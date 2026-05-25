"""
main.py — ocr-service (port 8003)
Pipeline YOLO → PaddleOCR pour extraction des champs CIN tunisienne.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routers.ocr import router

app = FastAPI(title="ocr-service — eKYC CIN", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup():
    logger.info("ocr-service demarre — warm-up modeles en background...")

    def _warmup():
        try:
            from services.pipeline import init_pipeline
            from routers.ocr import _pipeline_lock, _set_cin_initialized
            with _pipeline_lock:
                init_pipeline()
                _set_cin_initialized()
            logger.info("Warm-up pipeline CIN termine.")
        except Exception as e:
            logger.warning(f"Warm-up echoue (non bloquant): {e}")

    threading.Thread(target=_warmup, daemon=True).start()


@app.get("/health", tags=["Monitoring"])
def health():
    from services.service_yolo import is_loaded as yolo_ready, is_loaded_passport
    from services.service_ocr import is_loaded as ocr_ready
    return {
        "status": "ok",
        "service": "ocr-service",
        "port": 8003,
        "yolo_recto":    yolo_ready("recto"),
        "yolo_verso":    yolo_ready("verso"),
        "yolo_passport": is_loaded_passport(),
        "ocr_engine":    ocr_ready(),
    }
