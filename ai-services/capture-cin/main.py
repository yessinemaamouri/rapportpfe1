"""
main.py — capture-cin (port 8001)
Feedback caméra temps réel pour apps/client.
Appelé directement par le frontend via token éphémère signé par backend :8000.
Zéro persistance propre — MinIO/DB uniquement pour les captures confirmées.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path, override=True)

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routers.detection import router as detection_router, set_services
from services.minio_service import MinioService
from services.db_service import DbService

app = FastAPI(title="capture-cin — eKYC real-time feedback", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)


@app.on_event("startup")
def init_services() -> None:
    minio, db = None, None
    try:
        minio = MinioService()
        logger.info("MinIO service initialisé.")
    except Exception as e:
        logger.error(f"MinIO indisponible: {e}")
    try:
        db = DbService()
        db.init_table()
        logger.info("DB service initialisé.")
    except Exception as e:
        logger.error(f"PostgreSQL indisponible: {e}")
    set_services(minio, db)


@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "ok",
        "service": "capture-cin",
        "port": 8001,
        "gpu_available": torch.cuda.is_available() if hasattr(torch, "cuda") else False,
    }
