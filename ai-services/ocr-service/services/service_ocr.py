"""
services/service_ocr.py
PaddleOCR PP-OCRv5 — deux moteurs : arabe (texte) + français (chiffres latins).
"""
import re
import os
import cv2
import numpy as np
from loguru import logger

from .cin_post_processing import normalize_dob, postprocess_cin_field
from .post_ocr_passport import normalize_latin_date, postprocess_passport_rtl

OCR_VERSION: str = "PP-OCRv5"
OUTPUT_IMG_DIR: str = "outputimg"
OUTPUT_JSON_DIR: str = "outputjson"

# CIN : chiffres uniquement
_DIGITS_ONLY_LABELS: frozenset[str] = frozenset({"num_cin", "print_id"})

# CIN : mois arabe + chiffres
_MIXED_LABELS: frozenset[str] = frozenset({"dob", "issue_date"})

# Passeport : chiffres latins uniquement
_DIGITS_ONLY_LABELS_PASSPORT: frozenset[str] = frozenset({"num_pass", "num_cin"})

# Passeport : dates en chiffres latins purs
_MIXED_LABELS_PASSPORT: frozenset[str] = frozenset({"dob", "issue_date", "expiry_date"})

# Passeport : champs latin pur (last_name, first_name, pob en latin sur passeport)
_LATIN_LABELS_PASSPORT: frozenset[str] = frozenset({"last_name", "first_name", "pob"})

# Hauteur cible minimale par ligne de texte (px) avant d'envoyer à PaddleOCR.
# Augmenter cette valeur = upscale plus agressif = meilleure détection sur petits crops.
# Diminuer = moins d'upscale = plus rapide mais risque d'échec sur petits champs.
MIN_LINE_HEIGHT_PX: int = 80

# Nombre de lignes de texte par champ — utilisé pour calculer l'upscale adaptatif.
# Tous les champs sont sur 1 ligne sauf address du CIN verso (2 lignes).
_FIELD_LINE_COUNT: dict[str, int] = {
    "address": 2,  # CIN verso uniquement
}

# Champs sur fond texturé (filigrane, grisé) — binarisation adaptative appliquée
# pour séparer le texte foncé du fond avant envoi à PaddleOCR.
# Clé : doc_type ("cin" ou "passport"), valeur : ensemble de labels concernés.
_TEXTURED_BG_LABELS: dict[str, frozenset[str]] = {
    "cin":      frozenset({"profession"}),
    "passport": frozenset({"address"}),
}

_MAX_SIDE = 3800   # PaddleOCR devient très lent au-delà de 4000px

_ocr_ar = None
_ocr_en = None
# Moteurs sans correction d'orientation — utilisés pour les crops à fond texturé
# où use_doc_orientation_classify perturbe la détection de texte.
_ocr_ar_crop = None
_ocr_en_crop = None


def load_engine() -> None:
    global _ocr_ar, _ocr_en, _ocr_ar_crop, _ocr_en_crop
    if _ocr_ar is not None:
        return
    try:
        from paddleocr import PaddleOCR
        _ocr_ar = PaddleOCR(
            lang="ar",
            ocr_version=OCR_VERSION,
            use_doc_orientation_classify=True,
            use_textline_orientation=True,
            device="gpu",
        )
        _ocr_en = PaddleOCR(
            lang="fr",
            ocr_version=OCR_VERSION,
            use_doc_orientation_classify=True,
            use_textline_orientation=True,
            device="gpu",
        )
        _ocr_ar_crop = PaddleOCR(
            lang="ar",
            ocr_version=OCR_VERSION,
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
            device="gpu",
        )
        _ocr_en_crop = PaddleOCR(
            lang="fr",
            ocr_version=OCR_VERSION,
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
            device="gpu",
        )
        logger.info(f"PaddleOCR {OCR_VERSION} chargé (ar + fr, GPU).")
    except Exception as exc:
        logger.error(f"Échec chargement PaddleOCR : {exc}")
        raise


def is_loaded() -> bool:
    return _ocr_ar is not None


def _parse(ocr_result) -> tuple[list[str], list[float]]:
    if not ocr_result or not isinstance(ocr_result[0], dict):
        return [], []
    return (
        ocr_result[0].get("rec_texts", []),
        ocr_result[0].get("rec_scores", []),
    )


def _parse_with_boxes(ocr_result) -> tuple[list[str], list[float], list]:
    """Retourne textes, scores et bboxes pour tri spatial RTL."""
    if not ocr_result or not isinstance(ocr_result[0], dict):
        return [], [], []
    texts  = ocr_result[0].get("rec_texts", [])
    scores = ocr_result[0].get("rec_scores", [])
    boxes  = ocr_result[0].get("dt_polys", [])
    simple_boxes = []
    for box in boxes:
        try:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            simple_boxes.append((min(xs), min(ys), max(xs), max(ys)))
        except Exception:
            simple_boxes.append((0, 0, 0, 0))
    return texts, scores, simple_boxes


def _preprocess(img_bgr: np.ndarray, label: str,
                bbox_height: int | None = None) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    # Utilise la hauteur bbox YOLO originale si disponible — le crop après
    # expansion/padding est plus grand que la zone réelle du texte, ce qui
    # ferait croire que le texte est grand alors qu'il est minuscule.
    ref_height = bbox_height if bbox_height and bbox_height > 0 else h
    nb_lines = _FIELD_LINE_COUNT.get(label, 1)
    height_per_line = ref_height / nb_lines
    if height_per_line < MIN_LINE_HEIGHT_PX:
        scale = MIN_LINE_HEIGHT_PX / height_per_line
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LANCZOS4)

    # Cap pour éviter le ralentissement PaddleOCR sur gros crops
    h2, w2 = img_bgr.shape[:2]
    if max(h2, w2) > _MAX_SIDE:
        scale = _MAX_SIDE / max(h2, w2)
        img_bgr = cv2.resize(img_bgr, (int(w2 * scale), int(h2 * scale)),
                             interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _binarize(crop_rgb: np.ndarray) -> np.ndarray:
    """Binarisation adaptative sur fond texturé — retourne une image RGB."""
    img_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    block = max(11, int(img_bgr.shape[0] * 0.08) | 1)
    binarized = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block,
        C=15,
    )
    m = min(50, binarized.shape[0] // 6, binarized.shape[1] // 6)
    binarized[:m, :] = 255
    binarized[-m:, :] = 255
    binarized[:, :m] = 255
    binarized[:, -m:] = 255
    return cv2.cvtColor(cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)


def _result_is_weak(res) -> bool:
    """Retourne True si le résultat OCR est vide ou de mauvaise qualité (score < 0.5)."""
    if not res or not res[0].get("rec_texts"):
        return True
    scores = res[0].get("rec_scores", [])
    return bool(scores) and float(np.mean(scores)) < 0.5


# Pour le CIN (recto + verso), l'orientation du document est toujours correcte —
# use_doc_orientation_classify perturbe la reconnaissance en tournant inutilement les crops.
# Le passeport garde le moteur avec orientation (document présenté dans des angles variés).
def _predict_ar(crop_rgb: np.ndarray, label: str, idx: int, doc_type: str = "cin"):
    is_textured = label in _TEXTURED_BG_LABELS.get(doc_type, frozenset())
    use_no_orient = doc_type == "cin"
    try:
        res = _ocr_ar_crop.predict(crop_rgb) if use_no_orient else _ocr_ar.predict(crop_rgb)
        if _result_is_weak(res):
            res = _ocr_ar_crop.predict(_binarize(crop_rgb))
        if is_textured and _result_is_weak(res):
            res = _ocr_ar.predict(_binarize(crop_rgb))
        return res
    except Exception as exc:
        logger.error(f"Erreur OCR ar sur {label}_{idx} : {exc}")
        return None


def _predict_en(crop_rgb: np.ndarray, label: str, idx: int, doc_type: str = "cin"):
    is_textured = label in _TEXTURED_BG_LABELS.get(doc_type, frozenset())
    use_no_orient = doc_type == "cin"
    try:
        res = _ocr_en_crop.predict(crop_rgb) if use_no_orient else _ocr_en.predict(crop_rgb)
        if _result_is_weak(res):
            res = _ocr_en_crop.predict(_binarize(crop_rgb))
        if is_textured and _result_is_weak(res):
            res = _ocr_en.predict(_binarize(crop_rgb))
        return res
    except Exception as exc:
        logger.error(f"Erreur OCR fr sur {label}_{idx} : {exc}")
        return None



def _build_result(texts: list, scores: list) -> dict:
    full_text = " ".join(t.strip() for t in texts if t.strip())
    avg_score = round(float(__import__("numpy").mean(scores)) if scores else 0.0, 4)
    lines = [{"text": t.strip(), "score": round(float(s), 4)} for t, s in zip(texts, scores)]
    return {"ocr_text": full_text, "ocr_score": avg_score, "ocr_lines": lines}


def _decode_crop(crop_bytes: bytes, label: str, detection_index: int, position):
    nparr = __import__("numpy").frombuffer(crop_bytes, __import__("numpy").uint8)
    crop_bgr = __import__("cv2").imdecode(nparr, __import__("cv2").IMREAD_COLOR)
    if crop_bgr is None:
        logger.warning(f"Impossible de decoder le crop pour {label}_{detection_index}")
        return None, None
    bbox_height = (position.get("y2", 0) - position.get("y1", 0)) if position else None
    return _preprocess(crop_bgr, label, bbox_height=bbox_height), bbox_height


def run_ocr_raw(
    crop_bytes: bytes,
    label: str,
    detection_index: int,
    position=None,
    doc_type: str = "cin",
) -> dict:
    """OCR brut PaddleOCR sans post-traitement. Retourne {ocr_text, ocr_score, ocr_lines}."""
    if _ocr_ar is None:
        raise RuntimeError("PaddleOCR non charge.")

    crop_rgb, _ = _decode_crop(crop_bytes, label, detection_index, position)
    if crop_rgb is None:
        return {"ocr_text": None, "ocr_score": None, "ocr_lines": None}

    if doc_type == "passport":
        if label in _LATIN_LABELS_PASSPORT or label in _DIGITS_ONLY_LABELS_PASSPORT or label in _MIXED_LABELS_PASSPORT:
            res = _predict_en(crop_rgb, label, detection_index, doc_type)
            _save_outputs(res, label, detection_index)
            texts, scores = _parse(res)
        else:
            res = _predict_ar(crop_rgb, label, detection_index, doc_type)
            _save_outputs(res, label, detection_index)
            texts, scores = _parse(res)
        if not texts:
            return {"ocr_text": None, "ocr_score": None, "ocr_lines": None}
        return _build_result(texts, scores)

    if label in _DIGITS_ONLY_LABELS:
        res = _predict_en(crop_rgb, label, detection_index, doc_type)
        _save_outputs(res, label, detection_index)
        texts, scores = _parse(res)
        if not texts:
            return {"ocr_text": None, "ocr_score": None, "ocr_lines": None}
        return _build_result(texts, scores)

    if label in _MIXED_LABELS:
        res_ar = _predict_ar(crop_rgb, label, detection_index, doc_type)
        res_en = _predict_en(crop_rgb, label, detection_index, doc_type)
        _save_outputs(res_ar, label, detection_index)
        texts_ar, scores_ar = _parse(res_ar)
        texts_en, scores_en = _parse(res_en)
        all_texts  = [t for t in texts_ar + texts_en if t.strip()]
        all_scores = scores_ar + scores_en
        if not all_texts:
            return {"ocr_text": None, "ocr_score": None, "ocr_lines": None}
        return _build_result(all_texts, all_scores)

    res = _predict_ar(crop_rgb, label, detection_index, doc_type)
    _save_outputs(res, label, detection_index)
    texts, scores = _parse(res)
    if not texts:
        return {"ocr_text": None, "ocr_score": None, "ocr_lines": None}
    return _build_result(texts, scores)


def run_ocr(
    crop_bytes: bytes,
    label: str,
    detection_index: int,
    orig_img=None,
    position=None,
    doc_type: str = "cin",
) -> dict:
    """
    OCR complet = brut + post-traitement.
    Retourne {ocr_raw, ocr_text, ocr_score, ocr_lines}.
    ocr_raw  = texte brut PaddleOCR avant post-traitement.
    ocr_text = texte final apres post-traitement.
    """
    if _ocr_ar is None:
        raise RuntimeError("PaddleOCR non charge.")

    crop_rgb, _ = _decode_crop(crop_bytes, label, detection_index, position)
    if crop_rgb is None:
        return {"ocr_raw": None, "ocr_text": None, "ocr_score": None, "ocr_lines": None}

    # ── PASSEPORT ─────────────────────────────────────────────────────────────
    if doc_type == "passport":
        if label in _LATIN_LABELS_PASSPORT or label in _DIGITS_ONLY_LABELS_PASSPORT:
            res = _predict_en(crop_rgb, label, detection_index, doc_type)
            texts, scores = _parse(res)
            if not texts:
                return {"ocr_raw": None, "ocr_text": None, "ocr_score": None, "ocr_lines": None}
            result = _build_result(texts, scores)
            return {"ocr_raw": result["ocr_text"], **result}

        if label in _MIXED_LABELS_PASSPORT:
            res_en = _predict_en(crop_rgb, label, detection_index, doc_type)
            texts_en, scores_en = _parse(res_en)
            raw = _build_result(texts_en, scores_en)["ocr_text"] if texts_en else None
            post = normalize_latin_date(texts_en, scores_en)
            return {"ocr_raw": raw, **post}

        res = _predict_ar(crop_rgb, label, detection_index, doc_type)
        _save_outputs(res, label, detection_index)
        texts, scores, boxes = _parse_with_boxes(res)
        if not texts:
            texts, scores = _parse(res)
            boxes = []
        raw_text = " ".join(t.strip() for t in texts if t.strip()) or None
        texts, scores = postprocess_passport_rtl(label, texts, scores, boxes)
        if not texts:
            return {"ocr_raw": raw_text, "ocr_text": None, "ocr_score": None, "ocr_lines": None}
        return {"ocr_raw": raw_text, **_build_result(texts, scores)}

    # ── CIN ───────────────────────────────────────────────────────────────────
    if label in _DIGITS_ONLY_LABELS:
        res = _predict_en(crop_rgb, label, detection_index, doc_type)
        _save_outputs(res, label, detection_index)
        texts, scores = _parse(res)
        if not texts:
            return {"ocr_raw": None, "ocr_text": None, "ocr_score": None, "ocr_lines": None}
        result = _build_result(texts, scores)
        return {"ocr_raw": result["ocr_text"], **result}

    if label in _MIXED_LABELS:
        res_ar = _predict_ar(crop_rgb, label, detection_index, doc_type)
        res_en = _predict_en(crop_rgb, label, detection_index, doc_type)
        _save_outputs(res_ar, label, detection_index)
        texts_ar, scores_ar = _parse(res_ar)
        # Tri RTL sur les fragments numériques — sur une CIN le jour est à droite (x élevé)
        # et l'année est à gauche (x faible) ; sans tri, PaddleOCR les fusionne en "02605"
        texts_en_raw, scores_en_raw, boxes_en = _parse_with_boxes(res_en)
        if texts_en_raw and boxes_en:
            combined = sorted(zip(texts_en_raw, scores_en_raw, boxes_en),
                               key=lambda x: x[2][0], reverse=True)  # tri x_min desc = RTL
            texts_en  = [c[0] for c in combined]
            scores_en = [c[1] for c in combined]
        else:
            texts_en, scores_en = texts_en_raw, scores_en_raw
        raw_text = " ".join([t for t in texts_ar + texts_en if t.strip()]) or None
        post = normalize_dob(texts_ar, texts_en, scores_ar, scores_en)
        return {"ocr_raw": raw_text, **post}

    res = _predict_ar(crop_rgb, label, detection_index, doc_type)
    _save_outputs(res, label, detection_index)

    _RTL_MULTILINE = {"full_name", "address", "mother_name", "nom_complet", "nom_mere", "profession"}
    if label in _RTL_MULTILINE:
        texts, scores, boxes = _parse_with_boxes(res)
        if texts and len(texts) > 1 and boxes:
            combined = list(zip(texts, scores, boxes))
            combined.sort(key=lambda x: x[2][1])
            heights = [b[3] - b[1] for _, _, b in combined if b[3] > b[1]]
            median_h = sorted(heights)[len(heights) // 2] if heights else 40
            line_height = max(20, int(median_h * 0.6))
            lines_grouped, current_line = [], [combined[0]]
            for item in combined[1:]:
                if abs(item[2][1] - current_line[0][2][1]) <= line_height:
                    current_line.append(item)
                else:
                    lines_grouped.append(sorted(current_line, key=lambda x: x[2][0], reverse=True))
                    current_line = [item]
            lines_grouped.append(sorted(current_line, key=lambda x: x[2][0], reverse=True))
            combined = [item for line in lines_grouped for item in line]
            texts  = [c[0] for c in combined]
            scores = [c[1] for c in combined]
    else:
        texts, scores = _parse(res)

    if not texts:
        logger.debug(f"Aucun texte pour {label}_{detection_index}")
        return {"ocr_raw": None, "ocr_text": None, "ocr_score": None, "ocr_lines": None}

    raw_text = " ".join(t.strip() for t in texts if t.strip()) or None
    texts, scores = postprocess_cin_field(label, texts, scores)
    if not texts:
        return {"ocr_raw": raw_text, "ocr_text": None, "ocr_score": None, "ocr_lines": None}
    return {"ocr_raw": raw_text, **_build_result(texts, scores)}


def _save_outputs(ocr_result, label: str, idx: int) -> None:
    if not ocr_result:
        return
    try:
        crop_dir = f"{label}_{idx}"
        os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
        os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
        for res in ocr_result:
            res.save_to_img(f"{OUTPUT_IMG_DIR}/{crop_dir}")
            res.save_to_json(f"{OUTPUT_JSON_DIR}/{crop_dir}")
    except Exception as exc:
        logger.debug(f"Sauvegarde outputs PaddleOCR échouée : {exc}")
