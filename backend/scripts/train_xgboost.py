"""
train_xgboost.py — Train the XGBoost fraud detection model.

Connects to the claims database (new schema: doctor_notes, admin_billing_code,
total_charge, is_fraud, fraud_type), engineers features, trains an XGBClassifier,
evaluates performance, and exports model + encoders as .pkl artifacts.

Usage:
    cd backend
    python scripts/train_xgboost.py
"""

import sys
import os
import json
import logging

# Ensure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(SCRIPT_DIR, "..")
DB_PATH = os.path.join(BACKEND_DIR, "jkn_fraud.db")
ML_MODELS_DIR = os.path.join(BACKEND_DIR, "ml_models")


def load_data() -> pd.DataFrame:
    """Load claims data from SQLite (new schema) into a DataFrame."""
    import sqlite3

    if not os.path.exists(DB_PATH):
        logger.error("Database not found at %s", DB_PATH)
        logger.info("Generating synthetic training data instead...")
        return generate_synthetic_data()

    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT
            claim_id,
            doctor_notes,
            admin_billing_code,
            length_of_stay,
            total_charge,
            is_fraud,
            fraud_type
        FROM claims
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        logger.warning("No data in database — generating synthetic data.")
        return generate_synthetic_data()

    logger.info("Loaded %d records from database.", len(df))
    return df


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic training data for development/demo purposes.
    Creates a realistic mix of legitimate and fraudulent claims.
    """
    np.random.seed(42)
    logger.info("Generating %d synthetic claim records...", n_samples)

    diagnosis_codes = [
        "A09.0", "A09.9", "B20", "E11.9", "I10", "I21.0", "I50.9",
        "J18.9", "K35.2", "K80.0", "M17.1", "N18.6", "S72.0", "R50.9",
    ]

    records = []
    for i in range(n_samples):
        is_fraud = np.random.random() < 0.25  # 25% fraud rate

        if is_fraud:
            total_charge = np.random.uniform(30_000_000, 150_000_000)
            length_of_stay = np.random.randint(1, 5)
            n_codes = np.random.randint(2, 6)
            diag = np.random.choice(["I21.0", "S72.0", "B20", "N18.6", "A09.9"])
            notes = np.random.choice([
                "Diare ringan 1 hari", "Demam biasa", "Flu biasa",
                "Batuk ringan", "Pilek ringan",
            ])
            codes = [diag] + np.random.choice(
                ["99.29", "03.90", "93.90", "88.72", "96.04"], 
                size=min(n_codes - 1, 5), replace=False
            ).tolist()
            fraud_type = np.random.choice(["upcoding", "unbundling"])
        else:
            total_charge = np.random.uniform(500_000, 30_000_000)
            length_of_stay = np.random.randint(1, 10)
            n_codes = np.random.randint(1, 3)
            diag = np.random.choice(diagnosis_codes)
            notes = np.random.choice([
                "Kontrol rutin", "Rawat jalan biasa", "Pemeriksaan lengkap",
                "Keluhan standar", "Evaluasi berkala",
            ])
            codes = [diag]
            fraud_type = None

        records.append({
            "claim_id": f"CLM-{i+1:05d}",
            "doctor_notes": notes,
            "admin_billing_code": json.dumps(codes),
            "length_of_stay": length_of_stay,
            "total_charge": round(total_charge, 2),
            "is_fraud": int(is_fraud),
            "fraud_type": fraud_type,
        })

    return pd.DataFrame(records)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Feature engineering pipeline:
    - Parse admin_billing_code JSON
    - Label-encode primary billing code
    - Compute derived features (num_codes, cost_per_day, notes_length)
    """
    diag_encoder = LabelEncoder()
    proc_encoder = LabelEncoder()

    # Parse admin_billing_code JSON string into lists
    df["code_list"] = df["admin_billing_code"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )

    # Primary billing code (first code in the array)
    df["primary_code"] = df["code_list"].apply(
        lambda x: x[0] if x else "NONE"
    )

    # Derived features
    df["num_codes"] = df["code_list"].apply(len)
    df["cost_per_day"] = df["total_charge"] / df["length_of_stay"].clip(lower=1)
    df["notes_length"] = df["doctor_notes"].apply(len)

    # Label encoding
    df["primary_code_encoded"] = diag_encoder.fit_transform(df["primary_code"])
    # Use primary code for procedure encoding too (compatible with ml_repository)
    df["procedure_encoded"] = proc_encoder.fit_transform(df["primary_code"])

    feature_cols = [
        "length_of_stay",
        "total_charge",
        "primary_code_encoded",
        "procedure_encoded",
        "num_codes",
        "cost_per_day",
    ]

    # Rename to match ml_repository expectations
    df.rename(columns={
        "primary_code_encoded": "diagnosis_encoded",
    }, inplace=True)

    feature_cols = [
        "length_of_stay",
        "total_charge",
        "diagnosis_encoded",
        "procedure_encoded",
        "num_codes",
        "cost_per_day",
    ]

    logger.info("Feature engineering complete. Features: %s", feature_cols)
    return df, diag_encoder, proc_encoder


def train():
    """
    Full training pipeline:
    1. Load data from SQLite (or generate synthetic)
    2. Feature engineering
    3. Train/test split (80/20, stratified)
    4. Train XGBoost classifier
    5. Evaluate and print classification report
    6. Export model + encoders to ml_models/
    """
    logger.info("=" * 60)
    logger.info("KlaimGuard AI — XGBoost Training Pipeline")
    logger.info("=" * 60)

    # 1. Load data
    df = load_data()
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Fraud distribution:\n%s", df["is_fraud"].value_counts().to_string())

    # 2. Feature engineering
    df, diag_encoder, proc_encoder = engineer_features(df)

    feature_cols = [
        "length_of_stay",
        "total_charge",
        "diagnosis_encoded",
        "procedure_encoded",
        "num_codes",
        "cost_per_day",
    ]

    X = df[feature_cols].values
    y = df["is_fraud"].values

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train set: %d | Test set: %d", len(X_train), len(X_test))

    # 4. Train XGBoost
    pos_count = max(len(y_train[y_train == 1]), 1)
    neg_count = len(y_train[y_train == 0])

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_count / pos_count,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    logger.info("Training XGBClassifier...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # 5. Evaluate
    y_pred = model.predict(X_test)

    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logger.info("Accuracy: %.4f", accuracy)
    logger.info("F1-Score: %.4f", f1)

    # Feature importance
    logger.info("\nFeature Importance:")
    for col, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        logger.info("  %-25s %.4f", col, imp)

    # 6. Export artifacts
    os.makedirs(ML_MODELS_DIR, exist_ok=True)

    model_path = os.path.join(ML_MODELS_DIR, "xgboost_fraud.pkl")
    diag_path = os.path.join(ML_MODELS_DIR, "diagnosis_encoder.pkl")
    proc_path = os.path.join(ML_MODELS_DIR, "procedure_encoder.pkl")
    feat_path = os.path.join(ML_MODELS_DIR, "feature_columns.pkl")

    joblib.dump(model, model_path)
    joblib.dump(diag_encoder, diag_path)
    joblib.dump(proc_encoder, proc_path)
    joblib.dump(feature_cols, feat_path)

    logger.info("\nArtifacts saved to %s:", ML_MODELS_DIR)
    logger.info("  Model:              %s", model_path)
    logger.info("  Diagnosis encoder:  %s", diag_path)
    logger.info("  Procedure encoder:  %s", proc_path)
    logger.info("  Feature columns:    %s", feat_path)
    logger.info("=" * 60)
    logger.info("Training complete!")


if __name__ == "__main__":
    train()
