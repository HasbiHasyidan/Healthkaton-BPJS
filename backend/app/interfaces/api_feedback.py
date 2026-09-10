"""
Feedback API Router — endpoints for saving verifier feedback.

Endpoints per tambahan.md:
  POST /api/v1/feedback/submit  — Save verifier feedback to DB
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.infrastructure.database import get_db
from app.infrastructure.orm_models import FeedbackRecord
from app.domain.schemas import FeedbackInput

router = APIRouter()

@router.post("/v1/feedback/submit", tags=["Feedback"])
async def submit_feedback(feedback: FeedbackInput, db: Session = Depends(get_db)):
    """Save verifier feedback to DB."""
    try:
        new_feedback = FeedbackRecord(
            claim_id=feedback.claim_id,
            is_fraud=feedback.is_fraud,
            notes=feedback.notes
        )
        db.add(new_feedback)
        db.commit()
        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
