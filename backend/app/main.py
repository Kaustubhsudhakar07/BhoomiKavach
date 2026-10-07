"""
BHOOmiKAVACH AI — FastAPI Backend Application (Phase 23)
========================================================
Full Title:
  "BHOOmiKAVACH AI: An Explainable AI and Agentic Decision-Support
   System for Roof-Fall Risk Assessment and Prevention in Indian Mines"

Tagline:
  "SENSE THE RISK. EXPLAIN THE DANGER. PREVENT THE FALL."

Author: BHOOmiKAVACH AI Engineering Team
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.database.session import init_database_tables
from backend.app.api.routes import health, accidents, predict, rag, agent, audit


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database_tables()
    print(f"[{settings.PROJECT_NAME}] FastAPI Server initialized on {settings.API_V1_STR}")
    yield


app = FastAPI(
    title=f"{settings.PROJECT_NAME} — API Server",
    description="Explainable AI & Agentic Decision-Support System for DGMS Code 0111 Roof-Fall Risk Assessment.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["System Health & Diagnostics"])
app.include_router(accidents.router, prefix=f"{settings.API_V1_STR}/accidents", tags=["Historical Accidents"])
app.include_router(predict.router, prefix=f"{settings.API_V1_STR}/predict", tags=["Risk Prediction & Explainability"])
app.include_router(rag.router, prefix=f"{settings.API_V1_STR}/rag", tags=["Regulatory & Precedent RAG"])
app.include_router(agent.router, prefix=f"{settings.API_V1_STR}/agent", tags=["Agentic AI Decision Support"])
app.include_router(audit.router, prefix=f"{settings.API_V1_STR}/audit", tags=["Audit Trail"])


@app.get("/")
def root_redirect():
    return {
        "project": settings.PROJECT_NAME,
        "title": "BHOOmiKAVACH AI Decision-Support Platform",
        "tagline": "SENSE THE RISK. EXPLAIN THE DANGER. PREVENT THE FALL.",
        "documentation": "/docs",
        "api_v1": settings.API_V1_STR,
        "safety_disclaimer": "AI-assisted risk assessment only. Competent technical assessment is required."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
