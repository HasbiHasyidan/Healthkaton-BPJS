#!/bin/bash
set -e

echo "=== KlaimGuard AI Startup Script ==="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Exiting."
    exit 1
fi

# Check for virtual environment or create one
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize Database
echo "Initializing Database..."
cd backend
python scripts/init_db.py

# Train XGBoost model
echo "Training XGBoost model..."
python scripts/train_xgboost.py

# Start Server
echo "Starting Uvicorn Server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000
