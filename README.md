# JKN Fraud AI Agent — Copilot Verifikator Medis

Healthcare insurance fraud detection prototype for Healthkathon 2026.  
Automatically flags anomalous medical claims (upcoding & unbundling) using a hybrid Rule-Based + Machine Learning approach.

## Tech Stack

| Layer          | Technology                        |
|----------------|-----------------------------------|
| Backend        | Python 3.10+, FastAPI             |
| ML             | XGBoost, Scikit-Learn             |
| Database & ETL | SQLite, SQLAlchemy, Pandas        |
| Frontend       | HTML5, CSS3, Vanilla JavaScript   |

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize database
python backend/scripts/init_db.py

# 4. Train the model
python backend/scripts/train_xgboost.py

# 5. Run the API server
uvicorn backend.main:app --reload --port 8000
```

## Architecture

Clean Architecture with four layers:

- **Domain** — Pydantic schemas and business entities
- **Use Cases** — Fraud detection logic, model retraining orchestration
- **Interfaces** — FastAPI routers (API controllers)
- **Infrastructure** — Database connections, ORM models, ML artifact loading
