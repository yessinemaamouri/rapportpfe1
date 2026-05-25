"""
Détecteur de visage via YOLO ou Haar cascade (fallback).
Utilise le modèle face YOLO s'il est disponible, sinon cascade OpenCV.
"""
import os
import cv2
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH", "./yoloModels/face_best.pt")
YOLO_CONF       = float(os.getenv("FACE_YOLO_CONF", "0.40"))
VALID_CONF      = float(os.getenv("FACE_VALID_CONF", "0.60"))


class FaceDetector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        abs_path = self._resolve(FACE_MODEL_PATH)
        self._use_yolo = False
        self._model = None

        if os.path.exists(abs_path):
            try:
                from ultralytics import YOLO
                self._model = YOLO(abs_path)
                self._use_yolo = True
                logger.info(f"Face detector: modèle YOLO chargé depuis {abs_path}")
            except Exception as e:
                logger.warning(f"Face detector: YOLO indisponible ({e}), fallback Haar")

        if not self._use_yolo:
            # Fallback : cascade Haar d'OpenCV (intégrée dans le package)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Face detector: Haar cascade chargée")

    @staticmethod
    def _resolve(path: str) -> str:
        if os.path.isabs(path):
            return path
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(base, path)

    def detect(self, frame_bgr: np.ndarray) -> dict | None:
        """
        Retourne le meilleur visage détecté ou None.
        Dict: { bbox: [x1,y1,x2,y2], confidence: float, conf_ok: bool }
        """
        if self._use_yolo and self._model:
            return self._detect_yolo(frame_bgr)
        return self._detect_haar(frame_bgr)

    def _detect_yolo(self, frame: np.ndarray) -> dict | None:
        results = self._model.predict(frame, conf=YOLO_CONF, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None
        box  = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        return {"bbox": [x1, y1, x2, y2], "confidence": conf, "conf_ok": conf >= VALID_CONF}

    def _detect_haar(self, frame: np.ndarray) -> dict | None:
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if not len(faces):
            return None
        # Prend le plus grand visage
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        # Haar n'a pas de score → on retourne 0.75 comme proxy
        return {"bbox": [x, y, x + w, y + h], "confidence": 0.75, "conf_ok": True}
