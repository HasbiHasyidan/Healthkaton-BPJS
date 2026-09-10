"""
ML Repository — Singleton loader for XGBoost model and encoders.

Loads .pkl artifacts once at startup and provides predict_fraud_probability().
"""

import os
import logging
import numpy as np
import joblib
from typing import Optional

logger = logging.getLogger(__name__)

ML_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml_models")


class MLRepository:

    _instance: Optional["MLRepository"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = None
        self.diagnosis_encoder = None
        self.procedure_encoder = None
        self.feature_columns = None
        self._initialized = True
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all .pkl artifacts from the ml_models directory."""
        model_path = os.path.join(ML_MODELS_DIR, "xgboost_fraud.pkl")
        diag_enc_path = os.path.join(ML_MODELS_DIR, "diagnosis_encoder.pkl")
        proc_enc_path = os.path.join(ML_MODELS_DIR, "procedure_encoder.pkl")
        features_path = os.path.join(ML_MODELS_DIR, "feature_columns.pkl")

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info("XGBoost model loaded from %s", model_path)
        else:
            logger.warning("Model not found at %s — ML predictions disabled.", model_path)

        if os.path.exists(diag_enc_path):
            self.diagnosis_encoder = joblib.load(diag_enc_path)
            logger.info("Diagnosis encoder loaded.")

        if os.path.exists(proc_enc_path):
            self.procedure_encoder = joblib.load(proc_enc_path)
            logger.info("Procedure encoder loaded.")

        if os.path.exists(features_path):
            self.feature_columns = joblib.load(features_path)
            logger.info("Feature columns loaded: %d features", len(self.feature_columns))

    @property
    def is_ready(self) -> bool:
        """Check if the model and encoders are loaded and ready for inference."""
        return self.model is not None

    def predict_fraud_probability(self, features: dict) -> float:
        """
        Predict the upcoding fraud probability for a single claim.

        Args:
            features: dict with keys matching the trained feature set:
                - length_of_stay (int)
                - total_cost (float)
                - diagnosis_code (str)
                - procedure_codes (list[str])

        Returns:
            Probability (0.0–1.0) of the claim being upcoding fraud.
        """
        if not self.is_ready:
            logger.warning("ML model not loaded — returning default probability 0.0")
            return 0.0

        try:
            feature_vector = self._build_feature_vector(features)
            # predict_proba returns [[prob_class_0, prob_class_1]]
            proba = self.model.predict_proba(feature_vector)[0][1]
            return float(np.clip(proba, 0.0, 1.0))
        except Exception as e:
            logger.error("Prediction failed: %s", e)
            return 0.0

    def _build_feature_vector(self, features: dict) -> np.ndarray:
        """
        Transform raw claim features into the numeric vector expected by the model.

        Handles encoding of categorical variables using the fitted encoders.
        """
        length_of_stay = features.get("length_of_stay", 0)
        total_cost = features.get("total_cost", 0.0)
        diagnosis_code = features.get("diagnosis_code", "UNKNOWN")
        procedure_codes = features.get("procedure_codes", [])

        # Encode diagnosis code
        diag_encoded = 0
        if self.diagnosis_encoder is not None:
            try:
                diag_encoded = self.diagnosis_encoder.transform([diagnosis_code])[0]
            except ValueError:
                # Unseen label — use 0 (unknown)
                diag_encoded = 0

        # Encode procedure code — use the first code or "NONE"
        proc_encoded = 0
        if self.procedure_encoder is not None:
            primary_proc = procedure_codes[0] if procedure_codes else "NONE"
            try:
                proc_encoded = self.procedure_encoder.transform([primary_proc])[0]
            except ValueError:
                proc_encoded = 0

        # Derived features
        num_procedures = len(procedure_codes)
        cost_per_day = total_cost / max(length_of_stay, 1)

        feature_vector = np.array([[
            length_of_stay,
            total_cost,
            diag_encoded,
            proc_encoded,
            num_procedures,
            cost_per_day,
        ]])

        return feature_vector

    def save_model(self, model, diagnosis_encoder=None, procedure_encoder=None, feature_columns=None):
        """Persist trained model and encoders to disk."""
        os.makedirs(ML_MODELS_DIR, exist_ok=True)

        joblib.dump(model, os.path.join(ML_MODELS_DIR, "xgboost_fraud.pkl"))
        if diagnosis_encoder is not None:
            joblib.dump(diagnosis_encoder, os.path.join(ML_MODELS_DIR, "diagnosis_encoder.pkl"))
        if procedure_encoder is not None:
            joblib.dump(procedure_encoder, os.path.join(ML_MODELS_DIR, "procedure_encoder.pkl"))
        if feature_columns is not None:
            joblib.dump(feature_columns, os.path.join(ML_MODELS_DIR, "feature_columns.pkl"))

        # Reload into the singleton
        self.model = model
        self.diagnosis_encoder = diagnosis_encoder
        self.procedure_encoder = procedure_encoder
        self.feature_columns = feature_columns
        logger.info("Model and encoders saved to %s", ML_MODELS_DIR)


# ── Module-level singleton for import convenience ────────────
ml_repo = MLRepository()
