"""
Claims API Router — v1 endpoints for analyzing and batch-simulating claims.

Endpoints per prompt.md:
  POST /api/v1/claims/analyze         — Analyze a single claim
  POST /api/v1/claims/simulate-batch  — Simulate batch (accepts integer N)
"""

import json
import time
import random
import logging
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.domain.schemas import (
    ClaimAnalysisRequest,
    SimulateBatchRequest,
    AnalysisResult,
    BatchSimulationResponse,
)
from app.use_cases.fraud_detector import FraudDetector
from app.infrastructure.database import get_db
from app.infrastructure.orm_models import ClaimRecord

logger = logging.getLogger(__name__)

router = APIRouter()

# Instantiate use case (singleton — shares ml_repo internally)
detector = FraudDetector()


# ── POST /api/v1/claims/analyze ──────────────────────────────

@router.post("/v1/analyze", response_model=AnalysisResult)
async def analyze_claim(claim: ClaimAnalysisRequest):
    """
    Analyze a single claim using the hybrid fraud detection engine.

    Runs rule-based unbundling checks first, then ML-based upcoding detection.
    Returns a structured result with status (FLAGGED/SAFE), probability score,
    and explainable AI narrative in Bahasa Indonesia.
    """
    result = detector.analyze_claim(
        claim_id=claim.claim_id,
        doctor_notes=claim.doctor_notes,
        admin_billing_code=claim.admin_billing_code,
        length_of_stay=claim.length_of_stay,
        total_charge=claim.total_charge,
    )
    return result


# ── POST /api/v1/claims/simulate-batch ──────────────────────

@router.post("/v1/simulate-batch", response_model=BatchSimulationResponse)
async def simulate_batch(request: SimulateBatchRequest, db: Session = Depends(get_db)):
    """
    Simulate batch fraud detection (crucial for demo).

    Accepts an integer N. Randomly pulls N claims from the database,
    processes them through the AI engine in batch, and returns a summary.

    Example response summary: "Processed 1000 claims in 0.4s. 50 Flagged, 950 Safe"
    """
    n = request.n
    start_time = time.perf_counter()

    # Pull all claims from DB then randomly sample N
    all_claims = db.query(ClaimRecord).all()

    if not all_claims:
        return BatchSimulationResponse(
            total=0,
            flagged=0,
            safe=0,
            processing_time_ms=0,
            summary="Database kosong — jalankan init_db.py terlebih dahulu.",
            results=[],
        )

    # If N > available claims, sample with replacement
    if n <= len(all_claims):
        sampled = random.sample(all_claims, n)
    else:
        sampled = random.choices(all_claims, k=n)

    # Process each claim through the fraud detector
    results: List[AnalysisResult] = []
    for claim in sampled:
        # Parse admin_billing_code from JSON string
        try:
            billing_codes = json.loads(claim.admin_billing_code)
        except (json.JSONDecodeError, TypeError):
            billing_codes = []

        result = detector.analyze_claim(
            claim_id=claim.claim_id,
            doctor_notes=claim.doctor_notes,
            admin_billing_code=billing_codes,
            length_of_stay=claim.length_of_stay,
            total_charge=claim.total_charge,
        )
        results.append(result)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    flagged = sum(1 for r in results if r.status == "FLAGGED")
    safe = len(results) - flagged

    summary = (
        f"Processed {len(results)} claims in {elapsed_ms / 1000:.1f}s. "
        f"{flagged} Flagged, {safe} Safe"
    )

    return BatchSimulationResponse(
        total=len(results),
        flagged=flagged,
        safe=safe,
        processing_time_ms=round(elapsed_ms, 1),
        summary=summary,
        results=results,
    )
