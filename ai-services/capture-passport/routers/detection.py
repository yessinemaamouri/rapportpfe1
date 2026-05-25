"""
routers/detection.py
WebSocket feedback temps réel + endpoints HTTP pour la capture de passeport.
Passeport = capture unique (pas de recto/verso), cadre portrait (~3:4).
"""
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from schemas.detection import (
    DetectionRequest,
    DetectionResponse,
    CaptureRequest,
    CaptureResponse,
)
from services.detector import YoloDetector
from services.stabilizer import Stabilizer

router = APIRouter()

PASSPORT_MODEL_PATH = os.getenv("PASSPORT_MODEL_PATH", "./yoloModels/passport_best.pt")

detector_passport = YoloDetector(PASSPORT_MODEL_PATH)

minio_service = None


def set_services(minio, db=None):
    global minio_service
    minio_service = minio


# ── Utilitaires image ─────────────────────────────────────────────────────────

def base64_to_cv2(b64_string: str) -> np.ndarray:
    if "base64," in b64_string:
        b64_string = b64_string.split("base64,")[1]
    img_data = base64.b64decode(b64_string)
    nparr    = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def crop_with_padding(frame: np.ndarray, bbox: list[int], pad: int = 8) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    return frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]


def encode_jpeg_bytes(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
    if not ok:
        raise ValueError("Impossible d'encoder l'image en JPEG.")
    return encoded.tobytes()


def upscale_for_ocr(frame: np.ndarray, target_w: int = 1920, target_h: int = 1080) -> np.ndarray:
    h, w = frame.shape[:2]
    if w >= target_w and h >= target_h:
        return frame
    scale = max(target_w / w, target_h / h)
    temp  = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    temp  = cv2.filter2D(temp, -1, kernel)
    return cv2.resize(temp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    Flux vidéo temps réel: frame → YOLO passeport → feedback JSON.
    Status: ABSENT | DETECTED | CONFIRMED
    Auto-capture déclenchée par le Stabilizer (5 frames consécutives valides).
    """
    await websocket.accept()
    stabilizer = Stabilizer()
    stabilizer.reset()
    logger.info("WebSocket passeport ouvert.")

    try:
        while True:
            try:
                data = await websocket.receive_json()
                frame_detect_b64 = data.get("frame_detect") or data.get("frame")
                if not frame_detect_b64:
                    raise ValueError("frame manquante")
                req = DetectionRequest.model_validate({
                    "frame":  frame_detect_b64,
                    "guide":  data.get("guide"),
                    "screen": data.get("screen"),
                })
            except Exception as e:
                logger.debug(f"payload invalide: {e}")
                continue

            try:
                frame = base64_to_cv2(req.frame)
            except Exception as e:
                logger.debug(f"base64 invalide: {e}")
                continue

            detection = detector_passport.detect(frame, guide=req.guide.model_dump(), screen=req.screen.model_dump())

            if not detection:
                stabilizer.reset()
                resp = DetectionResponse(
                    status="ABSENT", confidence=None,
                    is_inside_guide=False, should_capture=False,
                )
            else:
                is_inside   = bool(detection["is_inside_guide"])
                conf        = float(detection["confidence"])
                is_valid    = bool(detection["conf_ok"] and is_inside)
                status, _progress, should_capture = stabilizer.update(is_valid)

                resp = DetectionResponse(
                    status="CONFIRMED" if status == "CONFIRMED" else "DETECTED",
                    confidence=conf,
                    is_inside_guide=is_inside,
                    should_capture=bool(should_capture),
                    debug=detection.get("debug"),
                )

                if should_capture and minio_service:
                    frame_capture_b64 = data.get("frame_capture")
                    if frame_capture_b64:
                        try:
                            frame_capture = base64_to_cv2(frame_capture_b64)
                            h_det, w_det  = frame.shape[:2]
                            h_cap, w_cap  = frame_capture.shape[:2]
                            bbox = detection.get("bbox")
                            if bbox and w_det > 0 and h_det > 0:
                                sx, sy = w_cap / w_det, h_cap / h_det
                                x1, y1, x2, y2 = bbox
                                bbox_hr = [
                                    max(0, min(w_cap-1, int(x1*sx))),
                                    max(0, min(h_cap-1, int(y1*sy))),
                                    max(0, min(w_cap,   int(x2*sx))),
                                    max(0, min(h_cap,   int(y2*sy))),
                                ]
                                x1h, y1h, x2h, y2h = bbox_hr
                                if x2h > x1h and y2h > y1h:
                                    crop = upscale_for_ocr(frame_capture[y1h:y2h, x1h:x2h])
                                    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
                                    if ok:
                                        capture_id = f"ws_passport_{int(time.time() * 1000)}"
                                        url = minio_service.upload_image(buf.tobytes(), capture_id)
                                        resp.capture_id = capture_id
                                        resp.image_url  = url
                        except Exception as e:
                            logger.debug(f"auto-capture passeport échouée: {e}")

            await websocket.send_json(resp.model_dump())

    except WebSocketDisconnect:
        logger.info("Client WebSocket passeport déconnecté.")
    except Exception as e:
        logger.error(f"WebSocket passeport rompu: {e}")


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@router.post("/capture-passport", tags=["Capture"])
def capture_passport_frame(request: CaptureRequest):
    """Capture manuelle du passeport déclenchée par le frontend."""
    logger.info(f"Capture passeport manuelle: {request.capture_id}")
    if not minio_service:
        return JSONResponse(status_code=503, content={"detail": "Service MinIO indisponible."})
    try:
        frame = base64_to_cv2(request.frame)
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Base64 corrompu."})

    h, w = frame.shape[:2]
    detection = detector_passport.detect(
        frame,
        guide={"x": 0, "y": 0, "width": w, "height": h},
        screen={"width": w, "height": h},
    )

    final_frame = frame
    confidence  = None
    if detection:
        confidence = float(detection["confidence"])
        bbox = detection.get("bbox")
        if bbox:
            final_frame = crop_with_padding(frame, bbox)
            final_frame = upscale_for_ocr(final_frame)

    jpeg_bytes = encode_jpeg_bytes(final_frame)
    url = minio_service.upload_image(jpeg_bytes, request.capture_id)
    logger.info(f"Passeport stocké: {request.capture_id} → {url}")
    return CaptureResponse(success=True, capture_id=request.capture_id, image_path=url)


class DetectRequest(BaseModel):
    image: str | None = None
    image_path: str | None = None


@router.post("/detect", tags=["Detection"])
async def detect_passport(request: DetectRequest):
    """
    Double usage:
    - Frontend caméra: { image: "<base64>" }
    - Pipeline Celery: { image_path: "bucket/key" }
    """
    if request.image_path:
        if getattr(detector_passport, "model", None) is None:
            raise HTTPException(status_code=503, detail="Modèle YOLO non chargé")
        try:
            return await detector_passport.detect_from_path(request.image_path)
        except Exception as e:
            logger.error(f"/detect passeport pipeline error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    if not request.image:
        raise HTTPException(status_code=422, detail="Fournir 'image' (base64) ou 'image_path' (MinIO)")

    try:
        frame = base64_to_cv2(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image base64 invalide: {e}")

    if getattr(detector_passport, "model", None) is None:
        return {"score": 0.0, "detected": False, "bbox": []}

    h, w = frame.shape[:2]
    detection = detector_passport.detect(
        frame,
        guide={"x": 0, "y": 0, "width": w, "height": h},
        screen={"width": w, "height": h},
    )

    if not detection:
        return {"score": 0.0, "detected": False, "bbox": []}

    conf = float(detection["confidence"])
    bbox = detection.get("bbox", [])
    return {"score": conf, "detected": conf >= 0.25, "bbox": bbox}
