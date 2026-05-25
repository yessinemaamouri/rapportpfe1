"""
capture-face — port 8007
Détection de visage temps réel + capture selfie.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routers.detection import router as detection_router, set_services
from services.minio_service import MinioService
from services.db_service import DbService

app = FastAPI(title="capture-face — eKYC selfie", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)


@app.on_event("startup")
def init_services():
    minio, db = None, None
    try:
        minio = MinioService()
        logger.info("MinIO service initialisé (capture-face).")
    except Exception as e:
        logger.error(f"MinIO indisponible: {e}")
    try:
        db = DbService()
        logger.info("DB service initialisé (capture-face).")
    except Exception as e:
        logger.error(f"PostgreSQL indisponible: {e}")
    set_services(minio, db)


@app.get("/health")
def health():
    return {"status": "ok", "service": "capture-face", "port": 8007}
