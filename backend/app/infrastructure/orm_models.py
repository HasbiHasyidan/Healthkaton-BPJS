"""
ORM Models — SQLAlchemy table definitions for claims and feedback.

Schema per prompt.md:
  id, claim_id, doctor_notes, admin_billing_code (JSON), length_of_stay,
  total_charge, is_fraud, fraud_type
"""

from sqlalchemy import Column, String, Float, Integer, Boolean, Text, DateTime  # pyrefly: ignore[missing-import]
from sqlalchemy.sql import func  # pyrefly: ignore[missing-import]

from app.infrastructure.database import Base


class ClaimRecord(Base):
    """Persisted claim record — matches prompt.md schema."""
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(String, unique=True, nullable=False, index=True)
    doctor_notes = Column(Text, nullable=False)           # e.g. "Demam biasa"
    admin_billing_code = Column(Text, nullable=False)     # JSON array e.g. '["R50.9"]'
    length_of_stay = Column(Integer, nullable=False)
    total_charge = Column(Float, nullable=False)
    is_fraud = Column(Boolean, default=False)
    fraud_type = Column(String, nullable=True)            # "upcoding" / "unbundling" / null
    created_at = Column(DateTime, server_default=func.now())


class FeedbackRecord(Base):
    """Verifier feedback on a claim."""
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(String, nullable=False, index=True)
    is_fraud = Column(Boolean, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
