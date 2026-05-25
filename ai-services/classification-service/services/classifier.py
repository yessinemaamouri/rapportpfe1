"""
services/classifier.py
Charge YOLOcls une seule fois et expose classify(image_bytes).
"""
import os
from loguru import logger

DOCUMENT_CLASSES = ["CIN_RECTO", "CIN_VERSO", "PASSPORT", "ATTESTATION_TRAVAIL", "FICHE_PAIE"]

_model = None


def load_model(model_path: str) -> None:
    global _model
    try:
        from ultralytics import YOLO
        _model = YOLO(model_path)
        logger.info(f"YOLOcls chargé: {model_path}")
    except Exception as e:
        logger.error(f"Échec chargement YOLOcls: {e}")
        _model = None


def is_loaded() -> bool:
    return _model is not None


def classify(image_bytes: bytes) -> dict:
    """
    Retourne {"detected_type": str, "confidence": float}.
    Lève RuntimeError si le modèle n'est pas chargé.
    """
    if _model is None:
        raise RuntimeError("Modèle YOLOcls non chargé")

    import numpy as np
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = _model.predict(frame, verbose=False)
    if not results or not hasattr(results[0], "probs") or results[0].probs is None:
        return {"detected_type": "UNKNOWN", "confidence": 0.0}

    probs = results[0].probs
    top_idx = int(probs.top1)
    confidence = float(probs.top1conf)

    detected_type = DOCUMENT_CLASSES[top_idx] if top_idx < len(DOCUMENT_CLASSES) else "UNKNOWN"
    return {"detected_type": detected_type, "confidence": confidence}
