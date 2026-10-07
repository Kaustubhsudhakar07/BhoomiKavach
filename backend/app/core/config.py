"""
BHOOmiKAVACH AI — Application Configuration (Phase 23)
======================================================
"""

import os
from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = "BHOOmiKAVACH AI"
    API_V1_STR: str = "/api/v1"
    DB_PATH: str = "data/processed/bhoomikavach.db"
    MODEL_VERSION: str = "v1.2.0-champion"
    DATASET_VERSION: str = "DGMS-2016-2022-CANONICAL-V1"
    FEATURE_VERSION: str = "FEAT-84-CMRI-ISM-V1"
    CORS_ORIGINS: list = ["*"]
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


settings = Settings()
