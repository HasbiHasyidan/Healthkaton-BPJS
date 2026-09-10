@echo off
title JKN Fraud AI Agent — Startup
echo ============================================================
echo   JKN Fraud AI Agent — Copilot Verifikator Medis
echo   Healthkathon 2026
echo ============================================================
echo.

cd /d "%~dp0"

:: Step 1: Check for virtual environment
if not exist "venv\" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/5] Virtual environment found.
)

:: Activate venv
call venv\Scripts\activate.bat

:: Step 2: Install dependencies
echo [2/5] Installing dependencies...
pip install -r requirements.txt --quiet

:: Step 3: Initialize database if not exists
if not exist "backend\jkn_fraud.db" (
    echo [3/5] Initializing database and seeding data...
    cd backend
    python scripts\init_db.py
    cd ..
) else (
    echo [3/5] Database already exists.
)

:: Step 4: Train model if not exists
if not exist "backend\ml_models\xgboost_fraud.pkl" (
    echo [4/5] Training XGBoost model...
    cd backend
    python scripts\train_xgboost.py
    cd ..
) else (
    echo [4/5] ML model already trained.
)

:: Step 5: Start the server
echo [5/5] Starting FastAPI server...
echo.
echo ============================================================
echo   Dashboard: http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo   Press Ctrl+C to stop the server
echo ============================================================
echo.
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
