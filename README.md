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

## Deploy Gratis Untuk Uji Coba

### Render melalui GitHub

Repository ini sudah dilengkapi `Dockerfile`, `docker-entrypoint.sh`, dan `render.yaml`.
Saat container pertama kali dijalankan, aplikasi otomatis membuat database synthetic claims
dan melatih model XGBoost. Tidak perlu mengunggah `backend/jkn_fraud.db`, folder `venv`, atau
file `.pkl` ke GitHub.

1. Buat repository baru di GitHub.
2. Dari folder `jkn_fraud_prototype`, jalankan:

```powershell
git init
git add .
git commit -m "Prepare KlaimGuard deployment"
git branch -M main
git remote add origin https://github.com/USERNAME/klaimguard-ai.git
git push -u origin main
```

3. Di Render pilih **New > Blueprint** dan pilih repository tersebut.
4. Render akan membaca `render.yaml`, membangun Docker image, lalu menyediakan URL publik.

Catatan: paket Free dapat sleep saat tidak digunakan dan database SQLite bersifat sementara.
Untuk demo atau presentasi ini cukup; data akan dibuat ulang jika instance di-reset.
