"""
services/service_yolo.py
YOLOv11m — détection des zones de la CIN tunisienne (recto + verso).
"""
import os
import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

CONF_THRESHOLD: float = 0.2
IOU_THRESHOLD: float = 0.10
CROP_PADDING: int = 40

SKIP_OCR_LABELS: dict[str, frozenset[str]] = {
    "recto":    frozenset({"flag1", "flag2", "image"}),
    "verso":    frozenset({"flag3", "finger_print", "barcode"}),
    "passport": frozenset({"image"}),
}

# Expansion de la bbox YOLO sur l'image SOURCE (récupère le texte coupé).
# YOLO sous-détecte souvent les bords des champs denses/petits : on élargit
# la boîte vers les vrais pixels avant le crop. Valeurs en fraction de la
# largeur/hauteur de la bbox détectée.
# Format : label -> (left, right, top, bottom)  — valeurs négatives = réduction

# Expansion spécifique CIN verso — address : étend à gauche, réduit à droite
# (YOLO inclut "العنوان" label à droite qu'on veut exclure du crop OCR)
_BBOX_EXPAND_VERSO: dict[str, tuple[float, float, float, float]] = {
    "address": (0.10, 0.0, 0.05, 0.05),
}

# Expansions spécifiques au passeport uniquement (labels partagés avec CIN mais layout différent)
_BBOX_EXPAND_PASSPORT: dict[str, tuple[float, float, float, float]] = {
    "issue_date":  (0.06, 0.45, 0.18, 0.30),
    "expiry_date": (0.06, 0.45, 0.18, 0.30),
    "dob":         (0.06, 0.45, 0.18, 0.30),
    "address":     (0.05, 0.0, 0.05, 0.05),
    "profession":  (0.35, 0.15, 0.20, 0.30),
    "full_name":   (0.05, 0.05, 0.05, 0.05),
}
# Expansion par défaut appliquée aux champs passeport sans règle spécifique.
_BBOX_EXPAND_DEFAULT: tuple[float, float, float, float] = (0.08, 0.08, 0.12, 0.15)

DEFAULT_MODEL_RECTO:    str = "models/modelYolo_11_fit_recto/best.pt"
DEFAULT_MODEL_VERSO:    str = "models/modeleYolo11_verso_fit/best.pt"
DEFAULT_MODEL_PASSPORT: str = "models/modelYolo_passport/best.pt"

_models: dict[str, YOLO | None] = {"recto": None, "verso": None, "passport": None}


def load_model(model_path: str, side: str = "recto") -> None:
    if side not in _models:
        raise ValueError(f"side doit être 'recto', 'verso' ou 'passport', reçu : {side}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle YOLO introuvable : {model_path}")
    try:
        model = YOLO(model_path)
        # Patch préventif contre "'Conv' object has no attribute 'bn'" :
        # certains modèles YOLOv11 sont exportés avec les couches Conv déjà fusionnées
        # (BN absorbé) mais Ultralytics tente de re-fusionner au premier predict().
        # On marque chaque Conv comme déjà fusionné pour court-circuiter fuse().
        try:
            for m in model.model.modules():
                if type(m).__name__ == "Conv" and not hasattr(m, "bn"):
                    m.bn = None   # signale "déjà fusionné" à Ultralytics
        except Exception as patch_exc:
            logger.warning(f"Patch Conv.bn [{side}] ignoré : {patch_exc}")
        _models[side] = model
        logger.info(f"YOLOv11m [{side}] chargé : {model_path}")
    except Exception as exc:
        logger.error(f"Échec chargement YOLO [{side}] : {exc}")
        raise


def is_loaded(side: str = "recto") -> bool:
    return _models.get(side) is not None


def _make_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
               label: str, h: int, w: int, side: str = "recto") -> np.ndarray:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if side == "passport":
        el, er, et, eb = _BBOX_EXPAND_PASSPORT.get(label, _BBOX_EXPAND_DEFAULT)
    elif side == "verso" and label in _BBOX_EXPAND_VERSO:
        el, er, et, eb = _BBOX_EXPAND_VERSO[label]
    else:
        el, er, et, eb = (0.0, 0.0, 0.0, 0.0)

    x1c = max(0, int(x1 - bw * el))
    x2c = min(w, int(x2 + bw * er))
    y1c = max(0, int(y1 - bh * et))
    y2c = min(h, int(y2 + bh * eb))
     #hna full_name tjm tbadel les param
    # Pour CIN full_name : prend les 3/4 hauts de la bbox
    # (le 1/4 bas contient "تاريخ الولادة XX" qui appartient au champ dob)
    if side != "passport" and label == "full_name":
        y2c = y1c + int((y2c - y1c) * 0.75)

    crop = img[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        raise ValueError(f"Crop vide pour label='{label}' bbox=({x1},{y1},{x2},{y2})")

    # 2. Bordure blanche : marge de respiration pour le détecteur de texte OCR.
    crop = cv2.copyMakeBorder(
        crop, CROP_PADDING, CROP_PADDING, CROP_PADDING, CROP_PADDING,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    return crop


def detect_and_crop(
    image_bytes: bytes,
    side: str = "recto",
    save_dir: str | None = "detectedimages",
) -> list[dict]:
    model = _models.get(side)
    if model is None:
        raise RuntimeError(f"YOLOv11 [{side}] non chargé — appelez load_model() d'abord.")

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Impossible de décoder l'image fournie.")

    h, w = img.shape[:2]

    raw = model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=1280,
        rect=True,
        agnostic_nms=False,
        verbose=False,
    )

    boxes = raw[0].boxes
    if len(boxes) == 0:
        logger.warning(f"Aucune boîte détectée par YOLO [{side}].")
        return []

    skip_labels = SKIP_OCR_LABELS.get(side, frozenset())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    detections: list[dict] = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        label = model.names[int(box.cls[0])]
        yolo_score = round(float(box.conf[0]), 4)
        skip_ocr = label in skip_labels

        try:
            crop = _make_crop(img, x1, y1, x2, y2, label, h, w, side=side)
        except ValueError as exc:
            logger.warning(str(exc))
            continue

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            logger.warning(f"Échec encodage PNG pour {label}_{i}")
            continue
        crop_bytes = buf.tobytes()

        crop_path: str | None = None
        if save_dir:
            crop_path = os.path.join(save_dir, f"{label}_{i}.png")
            with open(crop_path, "wb") as f:
                f.write(crop_bytes)

        detections.append({
            "label":      label,
            "yolo_score": yolo_score,
            "position":   {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_bytes": crop_bytes,
            "crop_path":  crop_path,
            "skip_ocr":   skip_ocr,
        })

    detections.sort(key=lambda d: d["position"]["y1"])
    logger.info(f"YOLO [{side}] : {len(detections)} détection(s).")
    return detections


def is_loaded_passport() -> bool:
    return _models.get("passport") is not None
