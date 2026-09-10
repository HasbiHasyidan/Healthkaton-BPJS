"""
Domain Schemas — Pydantic models for claims, predictions, and API contracts.

Aligned with prompt.md spec:
- doctor_notes, admin_billing_code (JSON array), length_of_stay, total_charge
"""

from typing import Optional, List, Dict
from enum import Enum
from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────

class FraudStatus(str, Enum):
    FLAGGED = "FLAGGED"
    SAFE = "SAFE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ── Request Schemas ──────────────────────────────────────────

class ClaimAnalysisRequest(BaseModel):
    """Single medical claim submitted for fraud evaluation (v1)."""
    claim_id: str = Field(..., description="Unique claim identifier")
    doctor_notes: str = Field(..., description="Doctor's clinical notes (e.g. 'Demam biasa')")
    admin_billing_code: List[str] = Field(
        default_factory=list,
        description="Admin billing codes (ICD-10/ICD-9-CM array)"
    )
    length_of_stay: int = Field(..., ge=0, description="Length of stay in days")
    total_charge: float = Field(..., ge=0, description="Total claimed cost (IDR)")


class SimulateBatchRequest(BaseModel):
    """Request for batch simulation — accepts an integer N."""
    n: int = Field(
        default=10,
        ge=1,
        le=5000,
        description="Number of claims to randomly pull from the database"
    )


class FeedbackInput(BaseModel):
    """Verifier feedback on a flagged claim."""
    claim_id: str
    is_fraud: bool = Field(..., description="Verifier's final verdict")
    notes: Optional[str] = None


# ── Response Schemas ─────────────────────────────────────────

class BillingCodeTranslation(BaseModel):
    """Single billing code with its Indonesian translation."""
    code: str
    description: str


class AnalysisResult(BaseModel):
    """Structured fraud analysis output with XAI narrative — Indonesian."""
    claim_id: str
    doctor_notes: str
    admin_billing_code: List[str]
    admin_billing_translated: List[BillingCodeTranslation]
    total_charge: float
    length_of_stay: int
    status: FraudStatus
    probability_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    fraud_type: Optional[str] = Field(None, description="upcoding / unbundling / None")
    reason_narrative: str = Field(..., description="Human-readable XAI narrative in Indonesian")
    engine_used: str = Field(..., description="RULE_BASED / ML / HYBRID")


class BatchSimulationResponse(BaseModel):
    """Response for batch claim simulation (v1 simulate-batch endpoint)."""
    total: int
    flagged: int
    safe: int
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    summary: str = Field(..., description="Human-readable summary string")
    results: List[AnalysisResult]
