"""
main.py — capture-passport (port 8006)
Feedback caméra temps réel pour la capture de passeport.
Même principe que capture-cin (port 8001) mais cadre portrait (3:4) et capture unique (pas de verso).
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

app = FastAPI(title="capture-passport — eKYC real-time feedback", version="0.1.0")

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
    minio = None
    try:
        minio = MinioService()
        logger.info("MinIO service initialisé.")
    except Exception as e:
        logger.error(f"MinIO indisponible: {e}")
    set_services(minio)


@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "ok",
        "service": "capture-passport",
        "port": 8006,
        "gpu_available": torch.cuda.is_available() if hasattr(torch, "cuda") else False,
    }
