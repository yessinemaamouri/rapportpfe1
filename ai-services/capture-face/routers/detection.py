"""
routers/detection.py — capture-face (port 8002)
POST /detect  → détection temps réel (score + bbox)
POST /capture → upload Minio + save DB
"""
import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from typing import Optional

from services.detector import FaceDetector
from schemas.detection import DetectRequest

router = APIRouter()

_detector     = FaceDetector()
minio_service = None
db_service    = None


def set_services(minio, db):
    global minio_service, db_service
    minio_service = minio
    db_service    = db


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _b64_to_cv2(b64: str) -> np.ndarray:
    if "base64," in b64:
        b64 = b64.split("base64,")[1]
    data = base64.b64decode(b64)
    arr  = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_jpeg(frame: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Encodage JPEG impossible")
    return buf.tobytes()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/detect")
async def detect_face(req: DetectRequest):
    """
    Reçoit une frame base64, retourne le score de détection.
    Utilisé par le frontend en boucle (300 ms) pour le feedback temps réel.
    """
    try:
        frame = _b64_to_cv2(req.image)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Image invalide: {e}"})

    result = _detector.detect(frame)
    if not result:
        return {"score": 0.0, "detected": False, "conf_ok": False}

    return {
        "score":    result["confidence"],
        "detected": True,
        "conf_ok":  result["conf_ok"],
    }


class CaptureRequest(BaseModel):
    frame_detect:  str
    frame_capture: str
    guide:         Optional[dict] = None   # ignoré, conservé pour compatibilité frontend
    screen:        Optional[dict] = None   # ignoré, conservé pour compatibilité frontend
    user_id:       Optional[str] = None
    demande_id:    Optional[str] = None
    save_to_db:    bool = True             # False = MinIO uniquement, pas de persistence DB


@router.post("/capture")
async def capture_selfie(req: CaptureRequest):
    """
    YOLO tourne sur frame_detect (640×640).
    Si le visage est dans le cadre guide → crop de la zone guide sur frame_capture (HD).
    Sinon → crop YOLO bbox scalé sur HD.
    """
    if not minio_service or not db_service:
        return JSONResponse(status_code=503, content={"detail": "Services indisponibles."})

    try:
        frame_det = _b64_to_cv2(req.frame_detect)
        frame_cap = _b64_to_cv2(req.frame_capture)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Image invalide: {e}"})

    result        = _detector.detect(frame_det)
    confidence    = result["confidence"] if result else None
    face_detected = result is not None

    if result and result.get("bbox"):
        bbox = result["bbox"]
        h_det, w_det = frame_det.shape[:2]
        h_cap, w_cap = frame_cap.shape[:2]
        sx, sy = w_cap / w_det, h_cap / h_det

        x1, y1, x2, y2 = bbox
        pad = 30
        frame_cap = frame_cap[
            max(0,     int(y1 * sy) - pad) : min(h_cap, int(y2 * sy) + pad),
            max(0,     int(x1 * sx) - pad) : min(w_cap, int(x2 * sx) + pad),
        ]

    jpeg_bytes = _encode_jpeg(frame_cap)
    capture_id = f"selfie_{req.user_id or 'anon'}_{int(time.time() * 1000)}"

    try:
        url = minio_service.upload_selfie(jpeg_bytes, capture_id)
    except Exception as e:
        logger.error(f"MinIO upload selfie: {e}")
        return JSONResponse(status_code=500, content={"detail": "Erreur upload."})

    if req.save_to_db:
        try:
            db_service.save_selfie(
                capture_id    = capture_id,
                minio_url     = url,
                confidence    = confidence,
                face_detected = face_detected,
                user_id       = req.user_id,
                demande_id    = req.demande_id,
            )
        except Exception as e:
            logger.error(f"DB save selfie: {e}")

    logger.info(f"Selfie capturé: {capture_id} face={face_detected} conf={confidence}")
    return {
        "success":        True,
        "capture_id":     capture_id,
        "selfie_url":     url,
        "confidence":     confidence,
        "face_detected":  face_detected,
    }
