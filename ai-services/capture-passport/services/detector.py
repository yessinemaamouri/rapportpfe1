"""
services/detector.py
Détecteur YOLOv8 pour passeport (cadre portrait ~3:4).
Même logique que capture-cin mais modèle dédié passeport.
"""
import os
import numpy as np
from ultralytics import YOLO
from loguru import logger
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


def _resolve_model_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    capture_passport_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.abspath(os.path.join(capture_passport_dir, path))


YOLO_PREDICT_CONF = 0.25
VALIDATION_CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640
GUIDE_TOLERANCE_PX = 30


class YoloDetector:
    _instances = {}

    def __new__(cls, model_path=str):
        model_path = _resolve_model_path(model_path)
        if model_path not in cls._instances:
            logger.info(f"Chargement modèle YOLOv8 passeport: {model_path}")
            instance = super(YoloDetector, cls).__new__(cls)
            try:
                instance.model = YOLO(model_path)
                logger.info("Modèle passeport chargé avec succès.")
            except Exception as e:
                logger.error(f"Impossible de charger le modèle: {e}")
                instance.model = None
            cls._instances[model_path] = instance
        return cls._instances[model_path]

    def __init__(self, model_path=str):
        self.model_path = _resolve_model_path(model_path)

    @staticmethod
    def _is_bbox_inside_guide(
        *,
        bbox: list[int],
        guide: dict,
        screen: dict,
        frame_shape: tuple[int, int],
        tolerance_px: int = GUIDE_TOLERANCE_PX,
    ) -> tuple[bool, dict]:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

        frame_h, frame_w = frame_shape
        screen_w = max(1, int(screen["width"]))
        screen_h = max(1, int(screen["height"]))

        scale_x = frame_w / screen_w
        scale_y = frame_h / screen_h

        guide_x1 = int(guide["x"] * scale_x)
        guide_y1 = int(guide["y"] * scale_y)
        guide_x2 = int((guide["x"] + guide["width"]) * scale_x)
        guide_y2 = int((guide["y"] + guide["height"]) * scale_y)

        check_x1 = bbox_x1 >= guide_x1 - tolerance_px
        check_y1 = bbox_y1 >= guide_y1 - tolerance_px
        check_x2 = bbox_x2 <= guide_x2 + tolerance_px
        check_y2 = bbox_y2 <= guide_y2 + tolerance_px
        is_inside = check_x1 and check_y1 and check_x2 and check_y2

        debug = {
            "bbox_raw": [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
            "frame_size": [frame_w, frame_h],
            "screen_received": [screen_w, screen_h],
            "guide_received": [int(guide["x"]), int(guide["y"]), int(guide["width"]), int(guide["height"])],
            "guide_converted": [guide_x1, guide_y1, guide_x2, guide_y2],
            "scale": [round(scale_x, 6), round(scale_y, 6)],
            "checks": {
                "bbox_x1 >= guide_x1-tol": check_x1,
                "bbox_y1 >= guide_y1-tol": check_y1,
                "bbox_x2 <= guide_x2+tol": check_x2,
                "bbox_y2 <= guide_y2+tol": check_y2,
            },
        }
        return is_inside, debug

    def detect(self, frame_bgr: np.ndarray, guide: dict, screen: dict) -> dict | None:
        if self.model is None:
            logger.warning("Modèle YOLO non chargé.")
            return None

        results = self.model.predict(
            frame_bgr,
            conf=YOLO_PREDICT_CONF,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        if not len(results) or len(results[0].boxes) == 0:
            return None

        best_box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        conf = float(best_box.conf[0])

        bbox = [x1, y1, x2, y2]
        frame_h, frame_w = frame_bgr.shape[:2]
        is_inside, debug = self._is_bbox_inside_guide(
            bbox=bbox,
            guide=guide,
            screen=screen,
            frame_shape=(frame_h, frame_w),
            tolerance_px=GUIDE_TOLERANCE_PX,
        )
        conf_ok = conf >= VALIDATION_CONF_THRESHOLD

        return {
            "bbox": bbox,
            "confidence": conf,
            "is_inside_guide": bool(is_inside),
            "conf_ok": bool(conf_ok),
            "debug": debug,
        }

    async def detect_from_path(self, image_path: str) -> dict:
        import os
        from minio import Minio

        endpoint   = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
        parts  = image_path.split("/", 1)
        bucket = parts[0] if len(parts) == 2 else "kyc-temp"
        key    = parts[1] if len(parts) == 2 else image_path

        response = client.get_object(bucket, key)
        try:
            image_bytes = response.read()
        finally:
            response.close()
            response.release_conn()

        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"is_valid": False, "confidence": 0.0, "document_detected": False, "reason": "Image illisible"}

        h, w = frame.shape[:2]
        full_guide  = {"x": 0, "y": 0, "width": w, "height": h}
        full_screen = {"width": w, "height": h}
        result = self.detect(frame, guide=full_guide, screen=full_screen)

        if result is None:
            return {"is_valid": False, "confidence": 0.0, "document_detected": False, "reason": "Aucun passeport détecté"}

        confidence = result["confidence"]
        is_valid   = bool(result["conf_ok"])
        reason     = None if is_valid else f"Confidence trop faible: {confidence:.2f}"
        return {"is_valid": is_valid, "confidence": confidence, "document_detected": True, "reason": reason}
