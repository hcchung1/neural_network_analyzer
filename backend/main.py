"""FastAPI Sidecar Entry Point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.sidecar import sidecar_router
from model_engine.transformer_model import get_model_engine

app = FastAPI(
    title="ArchAnalyzer Python Sidecar",
    description="Python backend sidecar for ArchAnalyzer Next.js app",
    version="0.2.0",
)

# CORS for internal communication if needed, though Next.js proxy avoids direct client calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Internal Sidecar APIs only
app.include_router(sidecar_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "sidecar": True, "model_loaded": get_model_engine().is_loaded}
