"""
KlaimGuard AI — FastAPI Application Entry Point

Serves both the API and the frontend dashboard from a single process.
1-click startup: uvicorn backend.main:app
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.interfaces.api_claims import router as claims_router
from app.interfaces.api_feedback import router as feedback_router


# ── Lifespan: load ML model at startup ───────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts when the server starts."""
    from app.infrastructure.ml_repository import ml_repo
    if ml_repo.is_ready:
        print("[OK] ML model loaded and ready for inference")
    else:
        print("[WARN] ML model not found -- running in rule-based mode only")
    yield


app = FastAPI(
    title="KlaimGuard AI",
    description="Copilot Verifikator Medis — Healthcare Fraud Detection API (Healthkathon 2026)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(claims_router, prefix="/api/claims", tags=["Claims"])
app.include_router(feedback_router, prefix="/api/feedback", tags=["Feedback"])


# ── Health check ─────────────────────────────────────────────

@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health-check endpoint."""
    from app.infrastructure.ml_repository import ml_repo
    return {
        "status": "ok",
        "service": "KlaimGuard AI",
        "ml_model_loaded": ml_repo.is_ready,
    }


# ── Serve frontend ──────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.isdir(FRONTEND_DIR):
    # Mount static assets (CSS, JS, images)
    app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")

    @app.get("/", tags=["Frontend"])
    async def serve_frontend():
        """Serve the dashboard at root."""
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    @app.get("/", tags=["Health"])
    async def root_fallback():
        return {"status": "ok", "service": "KlaimGuard AI", "note": "Frontend not found"}
