"""
init_db.py — Initialize the SQLite database, create tables, and seed 500+ synthetic claims.

Generates medically realistic data per prompt.md:
- SAFE claims: doctor notes match billing codes, reasonable charges
- UPCODING fraud: mild notes but severe billing codes + inflated charges
- UNBUNDLING fraud: single procedure notes but split billing codes

Usage:
    cd backend
    python scripts/init_db.py
"""

import sys
import os
import json
import random
import logging

# Ensure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.infrastructure.database import engine, Base, SessionLocal
from app.infrastructure.orm_models import ClaimRecord, FeedbackRecord  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Medical Knowledge Pools (Realistic Indonesian Healthcare)
# ══════════════════════════════════════════════════════════════

# ── SAFE claims: Notes match billing, charges are reasonable ──

SAFE_TEMPLATES = [
    {
        "notes": "Demam biasa, suhu 38.2°C, tidak ada komplikasi",
        "codes": ["R50.9"],
        "charge_range": (500_000, 3_000_000),
        "los_range": (1, 3),
    },
    {
        "notes": "ISPA ringan, batuk pilek 3 hari",
        "codes": ["J06.9"],
        "charge_range": (400_000, 2_500_000),
        "los_range": (1, 2),
    },
    {
        "notes": "Hipertensi terkontrol, kontrol rutin",
        "codes": ["I10"],
        "charge_range": (300_000, 2_000_000),
        "los_range": (1, 1),
    },
    {
        "notes": "Diabetes mellitus tipe 2, gula darah stabil",
        "codes": ["E11.9"],
        "charge_range": (500_000, 3_500_000),
        "los_range": (1, 3),
    },
    {
        "notes": "Gastroenteritis ringan, diare 2 hari, dehidrasi ringan",
        "codes": ["A09.0"],
        "charge_range": (800_000, 4_000_000),
        "los_range": (1, 3),
    },
    {
        "notes": "ISK tanpa komplikasi, nyeri berkemih",
        "codes": ["N39.0"],
        "charge_range": (600_000, 3_000_000),
        "los_range": (1, 3),
    },
    {
        "notes": "Nyeri punggung bawah, ketegangan otot",
        "codes": ["M54.5"],
        "charge_range": (400_000, 2_500_000),
        "los_range": (1, 2),
    },
    {
        "notes": "Nyeri perut akut, evaluasi lebih lanjut",
        "codes": ["R10.4"],
        "charge_range": (1_000_000, 5_000_000),
        "los_range": (1, 3),
    },
    {
        "notes": "Pneumonia komunitas, perlu rawat inap",
        "codes": ["J18.9", "87.44"],
        "charge_range": (5_000_000, 15_000_000),
        "los_range": (3, 7),
    },
    {
        "notes": "Apendisitis akut, operasi laparoskopi berhasil",
        "codes": ["K35.2", "47.01"],
        "charge_range": (10_000_000, 25_000_000),
        "los_range": (3, 5),
    },
    {
        "notes": "Persalinan normal tanpa komplikasi",
        "codes": ["O80"],
        "charge_range": (5_000_000, 12_000_000),
        "los_range": (2, 4),
    },
    {
        "notes": "Hiperlipidemia, konseling diet dan obat",
        "codes": ["E78.5"],
        "charge_range": (300_000, 1_500_000),
        "los_range": (1, 1),
    },
    {
        "notes": "Osteoarthritis lutut, fisioterapi rutin",
        "codes": ["M17.1"],
        "charge_range": (500_000, 3_000_000),
        "los_range": (1, 2),
    },
    {
        "notes": "Anemia, evaluasi dan suplementasi besi",
        "codes": ["D64.9", "89.52"],
        "charge_range": (800_000, 4_000_000),
        "los_range": (1, 3),
    },
    {
        "notes": "PPOK eksaserbasi, sesak napas, perlu nebulizer",
        "codes": ["J44.1"],
        "charge_range": (5_000_000, 18_000_000),
        "los_range": (3, 7),
    },
    {
        "notes": "Batu empedu, kolesistektomi elektif",
        "codes": ["K80.0", "51.22"],
        "charge_range": (12_000_000, 28_000_000),
        "los_range": (3, 6),
    },
    {
        "notes": "Epilepsi, kontrol rutin, obat stabil",
        "codes": ["G40.9"],
        "charge_range": (500_000, 2_500_000),
        "los_range": (1, 1),
    },
    {
        "notes": "Sesi kemoterapi rutin, kanker paru stadium III",
        "codes": ["Z51.1", "C34.9"],
        "charge_range": (15_000_000, 40_000_000),
        "los_range": (1, 3),
    },
    {
        "notes": "Gagal jantung kronik, evaluasi rutin + EKG",
        "codes": ["I50.9", "89.52"],
        "charge_range": (3_000_000, 12_000_000),
        "los_range": (2, 5),
    },
    {
        "notes": "Demam tifoid, perlu rawat inap dan antibiotik",
        "codes": ["A01.0"],
        "charge_range": (3_000_000, 8_000_000),
        "los_range": (3, 7),
    },
]

# ── UPCODING fraud: Mild notes + severe billing + inflated charges ──

UPCODING_TEMPLATES = [
    {
        "notes": "Diare ringan 1 hari",
        "codes": ["A09.9"],
        "charge_range": (12_000_000, 25_000_000),
        "los_range": (1, 2),
    },
    {
        "notes": "Demam biasa 2 hari",
        "codes": ["I21.0"],
        "charge_range": (30_000_000, 80_000_000),
        "los_range": (1, 3),
    },
    {
        "notes": "Batuk ringan, flu biasa",
        "codes": ["C34.9"],
        "charge_range": (50_000_000, 150_000_000),
        "los_range": (1, 3),
    },
    {
        "notes": "Pilek ringan, hidung tersumbat",
        "codes": ["B20"],
        "charge_range": (40_000_000, 120_000_000),
        "los_range": (1, 2),
    },
    {
        "notes": "Sakit kepala ringan, mual sedikit",
        "codes": ["I50.9"],
        "charge_range": (25_000_000, 70_000_000),
        "los_range": (1, 3),
    },
    {
        "notes": "Pusing ringan, istirahat cukup",
        "codes": ["N18.6"],
        "charge_range": (50_000_000, 180_000_000),
        "los_range": (1, 3),
    },
    {
        "notes": "Diare ringan anak, dehidrasi ringan",
        "codes": ["A09.9"],
        "charge_range": (15_000_000, 35_000_000),
        "los_range": (1, 2),
    },
    {
        "notes": "ISPA ringan, batuk kering",
        "codes": ["S72.0"],
        "charge_range": (60_000_000, 200_000_000),
        "los_range": (1, 5),
    },
    {
        "notes": "Flu biasa, demam 37.5°C",
        "codes": ["I21.0"],
        "charge_range": (35_000_000, 90_000_000),
        "los_range": (1, 2),
    },
    {
        "notes": "Mual ringan, tidak ada keluhan lain",
        "codes": ["C34.9"],
        "charge_range": (45_000_000, 130_000_000),
        "los_range": (1, 2),
    },
    {
        "notes": "Persalinan normal",
        "codes": ["O14.9"],
        "charge_range": (20_000_000, 50_000_000),
        "los_range": (3, 5),
    },
    {
        "notes": "Iritasi mata",
        "codes": ["13.41"],
        "charge_range": (10_000_000, 20_000_000),
        "los_range": (1, 2),
    },
    {
        "notes": "Batuk pilek biasa",
        "codes": ["J20.9"],
        "charge_range": (5_000_000, 15_000_000),
        "los_range": (1, 3),
    },
]

# ── UNBUNDLING fraud: Single procedure notes + split billing codes ──

UNBUNDLING_TEMPLATES = [
    {
        "notes": "Operasi Caesar, ibu dan bayi sehat",
        "codes": ["O80", "03.90", "93.90"],
        "charge_range": (30_000_000, 80_000_000),
        "los_range": (3, 7),
    },
    {
        "notes": "Operasi Caesar elektif",
        "codes": ["74.1", "03.90", "93.90"],
        "charge_range": (35_000_000, 90_000_000),
        "los_range": (3, 7),
    },
    {
        "notes": "Pemasangan stent jantung (PTCA)",
        "codes": ["36.01", "99.29", "88.72"],
        "charge_range": (50_000_000, 120_000_000),
        "los_range": (3, 10),
    },
    {
        "notes": "Operasi usus buntu laparoskopi",
        "codes": ["47.01", "99.23", "96.04"],
        "charge_range": (20_000_000, 60_000_000),
        "los_range": (3, 7),
    },
    {
        "notes": "Penggantian sendi panggul total",
        "codes": ["81.51", "03.09", "93.54"],
        "charge_range": (40_000_000, 100_000_000),
        "los_range": (5, 14),
    },
    {
        "notes": "Operasi patah tulang femur",
        "codes": ["79.31", "93.54", "04.81"],
        "charge_range": (30_000_000, 80_000_000),
        "los_range": (5, 14),
    },
    {
        "notes": "Pengangkatan empedu (kolesistektomi)",
        "codes": ["51.22", "99.29", "88.01"],
        "charge_range": (25_000_000, 70_000_000),
        "los_range": (3, 7),
    },
    {
        "notes": "Operasi Caesar darurat, komplikasi ringan",
        "codes": ["O80", "03.90", "93.90", "99.29"],
        "charge_range": (40_000_000, 110_000_000),
        "los_range": (4, 10),
    },
    {
        "notes": "Tindakan PTCA dengan evaluasi jantung",
        "codes": ["36.01", "99.29", "88.72", "89.52"],
        "charge_range": (60_000_000, 150_000_000),
        "los_range": (3, 10),
    },
    {
        "notes": "Apendektomi darurat, perlu transfusi",
        "codes": ["47.01", "99.23", "96.04", "99.04"],
        "charge_range": (25_000_000, 75_000_000),
        "los_range": (4, 10),
    },
    {
        "notes": "Cuci darah rutin",
        "codes": ["39.95", "99.21"],
        "charge_range": (5_000_000, 15_000_000),
        "los_range": (1, 2),
    },
    {
        "notes": "Operasi usus buntu",
        "codes": ["47.09", "03.90"],
        "charge_range": (15_000_000, 40_000_000),
        "los_range": (3, 7),
    },
]


def generate_claim(idx: int, template: dict, fraud_type: str | None) -> dict:
    """Generate a single claim from a template."""
    charge = round(random.uniform(*template["charge_range"]), 2)
    los = random.randint(*template["los_range"])

    return {
        "claim_id": f"CLM-{idx:05d}",
        "doctor_notes": template["notes"],
        "admin_billing_code": json.dumps(template["codes"]),
        "length_of_stay": los,
        "total_charge": charge,
        "is_fraud": fraud_type is not None,
        "fraud_type": fraud_type,
    }


def seed_data(session, n_total: int = 1200):
    """
    Seed the database with synthetic claim records.

    Distribution:
        ~70% legitimate (SAFE) claims
        ~15% upcoding fraud
        ~15% unbundling fraud
    """
    random.seed(42)
    np.random.seed(42)

    n_safe = int(n_total * 0.70)
    n_upcoding = int(n_total * 0.15)
    n_unbundling = n_total - n_safe - n_upcoding

    claims = []
    idx = 1

    # Generate SAFE claims
    for _ in range(n_safe):
        template = random.choice(SAFE_TEMPLATES)
        claims.append(generate_claim(idx, template, fraud_type=None))
        idx += 1

    # Generate UPCODING fraud
    for _ in range(n_upcoding):
        template = random.choice(UPCODING_TEMPLATES)
        claims.append(generate_claim(idx, template, fraud_type="upcoding"))
        idx += 1

    # Generate UNBUNDLING fraud
    for _ in range(n_unbundling):
        template = random.choice(UNBUNDLING_TEMPLATES)
        claims.append(generate_claim(idx, template, fraud_type="unbundling"))
        idx += 1

    # Shuffle to mix fraud with legitimate
    random.shuffle(claims)

    # Insert claim records
    claim_records = [ClaimRecord(**c) for c in claims]
    session.add_all(claim_records)
    session.flush()

    # Insert feedback for some claims
    feedback_count = 0
    fraud_claims = [c for c in claims if c["fraud_type"] is not None]
    safe_claims_list = [c for c in claims if c["fraud_type"] is None]

    # ~50% of fraud claims get confirmed feedback
    for c in fraud_claims:
        if random.random() < 0.5:
            fb = FeedbackRecord(
                claim_id=c["claim_id"],
                is_fraud=True,
                notes=f"Confirmed {c['fraud_type']} fraud — verified by auditor",
            )
            session.add(fb)
            feedback_count += 1

    # ~10% of safe claims get "not fraud" feedback
    for c in random.sample(safe_claims_list, min(30, len(safe_claims_list))):
        fb = FeedbackRecord(
            claim_id=c["claim_id"],
            is_fraud=False,
            notes="Reviewed and confirmed legitimate claim",
        )
        session.add(fb)
        feedback_count += 1

    session.commit()

    # Print summary
    logger.info("=" * 60)
    logger.info("SEEDING SUMMARY")
    logger.info("=" * 60)
    logger.info("Total claims:       %d", n_total)
    logger.info("  Safe/Legitimate:  %d (%.0f%%)", n_safe, n_safe / n_total * 100)
    logger.info("  Upcoding fraud:   %d (%.0f%%)", n_upcoding, n_upcoding / n_total * 100)
    logger.info("  Unbundling fraud: %d (%.0f%%)", n_unbundling, n_unbundling / n_total * 100)
    logger.info("Feedback records:   %d", feedback_count)
    logger.info("=" * 60)


def init_database():
    """Create all tables and seed with synthetic data."""
    logger.info("Creating database tables...")

    # Drop existing tables to start fresh with new schema
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created successfully.")

    session = SessionLocal()
    try:
        logger.info("Seeding database with 1200 synthetic claims...")
        seed_data(session, n_total=1200)
        logger.info("Database initialization complete!")
    finally:
        session.close()


if __name__ == "__main__":
    init_database()
