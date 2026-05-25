"""
routers/detection.py
WebSocket feedback temps réel + endpoints HTTP de capture.
Extrait de main.py pour respecter la séparation routers/services/schemas.
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

# ── Modèles chargés une seule fois ───────────────────────────────────────────
RECTO_MODEL_PATH = os.getenv("RECTO_MODEL_PATH", "./yoloModels/cin_recto_best.pt")
VERSO_MODEL_PATH = os.getenv("VERSO_MODEL_PATH", "./yoloModels/cin_verso_.pt")

detector_recto = YoloDetector(RECTO_MODEL_PATH)
detector_verso = YoloDetector(VERSO_MODEL_PATH)

# Injectés depuis main.py au démarrage
minio_service = None
db_service = None


def set_services(minio, db):
    global minio_service, db_service
    minio_service = minio
    db_service = db


def get_detector(model_type: str | None) -> YoloDetector:
    if model_type == "verso":
        return detector_verso
    return detector_recto


# ── Utilitaires image ─────────────────────────────────────────────────────────

def base64_to_cv2(b64_string: str) -> np.ndarray:
    if "base64," in b64_string:
        b64_string = b64_string.split("base64,")[1]
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
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
    temp = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    temp = cv2.filter2D(temp, -1, kernel)
    return cv2.resize(temp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    Flux vidéo temps réel: frame → YOLO → feedback JSON.
    Status: ABSENT | DETECTED | CONFIRMED
    Auto-capture déclenchée par le Stabilizer (5 frames consécutives valides).
    """
    await websocket.accept()
    stabilizer = Stabilizer()
    stabilizer.reset()
    logger.info("WebSocket ouvert — flux KYC engagé.")

    try:
        while True:
            try:
                data = await websocket.receive_json()
                frame_detect_b64 = data.get("frame_detect") or data.get("frame")
                if not frame_detect_b64:
                    raise ValueError("frame manquante")
                req = DetectionRequest.model_validate({
                    "frame": frame_detect_b64,
                    "guide": data.get("guide"),
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

            detector = get_detector(data.get("model", "recto"))
            detection = detector.detect(frame, guide=req.guide.model_dump(), screen=req.screen.model_dump())

            if not detection:
                stabilizer.reset()
                resp = DetectionResponse(
                    status="ABSENT", confidence=None,
                    is_inside_guide=False, should_capture=False,
                )
            else:
                is_inside = bool(detection["is_inside_guide"])
                conf = float(detection["confidence"])
                is_valid = bool(detection["conf_ok"] and is_inside)
                status, _progress, should_capture = stabilizer.update(is_valid)

                resp = DetectionResponse(
                    status="CONFIRMED" if status == "CONFIRMED" else "DETECTED",
                    confidence=conf,
                    is_inside_guide=is_inside,
                    should_capture=bool(should_capture),
                    debug=detection.get("debug"),
                )

                if should_capture and minio_service and db_service:
                    frame_capture_b64 = data.get("frame_capture")
                    if frame_capture_b64:
                        try:
                            frame_capture = base64_to_cv2(frame_capture_b64)
                            h_det, w_det = frame.shape[:2]
                            h_cap, w_cap = frame_capture.shape[:2]
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
                                        capture_id = f"ws_cin_{int(time.time() * 1000)}"
                                        url = minio_service.upload_image(buf.tobytes(), capture_id)
                                        db_service.save_capture(capture_id, url, conf)
                                        resp.capture_id = capture_id
                                        resp.image_url = url
                        except Exception as e:
                            logger.debug(f"auto-capture failed: {e}")

            await websocket.send_json(resp.model_dump())

    except WebSocketDisconnect:
        logger.info("Client WebSocket déconnecté.")
    except Exception as e:
        logger.error(f"WebSocket rompu: {e}")


# ── HTTP endpoints ────────────────────────────────────────────────────────────

class CaptureBothRequest(BaseModel):
    recto: str
    verso: str
    session_id: str


@router.post("/capture-both", tags=["Capture"])
def capture_both(req: CaptureBothRequest):
    if not minio_service:
        logger.error("capture-both: MinIO service non initialisé")
        return JSONResponse(status_code=503, content={"detail": "MinIO indisponible."})
    if not db_service:
        logger.error("capture-both: DB service non initialisé")
        return JSONResponse(status_code=503, content={"detail": "DB indisponible."})
    try:
        recto_b64 = req.recto.split("base64,")[1] if "base64," in req.recto else req.recto
        verso_b64 = req.verso.split("base64,")[1] if "base64," in req.verso else req.verso

        recto_bytes = base64.b64decode(recto_b64)
        verso_bytes = base64.b64decode(verso_b64)
        logger.info(f"capture-both: recto={len(recto_bytes)}B verso={len(verso_bytes)}B session={req.session_id}")

        url_r = minio_service.upload_image(recto_bytes, f"{req.session_id}_recto")
        url_v = minio_service.upload_image(verso_bytes, f"{req.session_id}_verso")
        logger.info(f"capture-both: MinIO OK recto={url_r}")

        db_service.save_capture(f"{req.session_id}_recto", url_r, 1.0)
        db_service.save_capture(f"{req.session_id}_verso", url_v, 1.0)
        logger.info(f"capture-both: DB OK session={req.session_id}")

        return {"success": True, "recto_url": url_r, "verso_url": url_v, "session_id": req.session_id}
    except Exception as e:
        logger.exception(f"capture-both ERREUR: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.post("/capture", response_model=CaptureResponse, tags=["Capture"])
def capture_cin_frame(request: CaptureRequest):
    """Capture déclenchée manuellement par le frontend."""
    logger.info(f"Capture manuelle: {request.capture_id}")
    if not minio_service or not db_service:
        return JSONResponse(status_code=503, content={"detail": "Services MinIO/DB indisponibles."})
    try:
        frame = base64_to_cv2(request.frame)
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Base64 corrompu."})

    h, w = frame.shape[:2]
    detector = get_detector("recto")
    detection = detector.detect(frame, guide={"x": 0, "y": 0, "width": w, "height": h}, screen={"width": w, "height": h})

    final_frame = frame
    confidence = None
    if detection:
        confidence = float(detection["confidence"])
        bbox = detection.get("bbox")
        if bbox:
            final_frame = crop_with_padding(frame, bbox)
            final_frame = upscale_for_ocr(final_frame)

    jpeg_bytes = encode_jpeg_bytes(final_frame)
    url = minio_service.upload_image(jpeg_bytes, request.capture_id)
    db_service.save_capture(request.capture_id, url, confidence)
    logger.info(f"Capture stockée: {request.capture_id} → {url}")
    return CaptureResponse(success=True, capture_id=request.capture_id, image_path=url)


class DetectRequest(BaseModel):
    # Frontend real-time mode: base64 frame
    image: str | None = None
    model: str | None = "recto"
    # Pipeline (Celery) mode: MinIO path
    image_path: str | None = None


@router.post("/detect", tags=["Detection"])
async def detect_document(request: DetectRequest):
    """
    Double usage:
    - Frontend caméra: { image: "<base64>", model: "recto"|"verso" }
      → retourne { score, detected, bbox }
    - Pipeline Celery: { image_path: "bucket/key" }
      → retourne { is_valid, confidence, document_detected, reason }
    """
    # ── Mode pipeline (Celery) ────────────────────────────────────────────────
    if request.image_path:
        detector = get_detector("recto")
        if getattr(detector, "model", None) is None:
            raise HTTPException(status_code=503, detail="Modèle YOLO non chargé")
        try:
            result = await detector.detect_from_path(request.image_path)
            return result
        except Exception as e:
            logger.error(f"/detect pipeline error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── Mode frontend (base64) ────────────────────────────────────────────────
    if not request.image:
        raise HTTPException(status_code=422, detail="Fournir 'image' (base64) ou 'image_path' (MinIO)")

    try:
        frame = base64_to_cv2(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image base64 invalide: {e}")

    detector = get_detector(request.model)
    if getattr(detector, "model", None) is None:
        # Modèle non chargé → score 0, pas de détection
        return {"score": 0.0, "detected": False, "bbox": []}

    h, w = frame.shape[:2]
    full_guide = {"x": 0, "y": 0, "width": w, "height": h}
    full_screen = {"width": w, "height": h}
    detection = detector.detect(frame, guide=full_guide, screen=full_screen)

    if not detection:
        return {"score": 0.0, "detected": False, "bbox": []}

    conf = float(detection["confidence"])
    bbox = detection.get("bbox", [])
    return {"score": conf, "detected": conf >= 0.25, "bbox": bbox}
