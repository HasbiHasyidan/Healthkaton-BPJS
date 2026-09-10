#!/bin/sh
set -eu

cd /app/backend

if [ ! -f "jkn_fraud.db" ]; then
    echo "[startup] Initializing synthetic claims database..."
    python scripts/init_db.py
else
    echo "[startup] Database already exists."
fi

if [ ! -f "ml_models/xgboost_fraud.pkl" ]; then
    echo "[startup] Training XGBoost model..."
    python scripts/train_xgboost.py
else
    echo "[startup] ML model already exists."
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
