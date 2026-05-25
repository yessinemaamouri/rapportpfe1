"""
services/pipeline.py
Orchestrateur du pipeline KYC CIN tunisienne — recto et verso.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from services.service_yolo import (
        load_model, is_loaded as yolo_ready, detect_and_crop,
        DEFAULT_MODEL_RECTO, DEFAULT_MODEL_VERSO, DEFAULT_MODEL_PASSPORT,
        is_loaded_passport,
    )
    from services.service_ocr import load_engine, is_loaded as ocr_ready, run_ocr
    from services.utils.report import create_final_report
else:
    from .service_yolo import (
        load_model, is_loaded as yolo_ready, detect_and_crop,
        DEFAULT_MODEL_RECTO, DEFAULT_MODEL_VERSO, DEFAULT_MODEL_PASSPORT,
        is_loaded_passport,
    )
    from .service_ocr import load_engine, is_loaded as ocr_ready, run_ocr
    from .utils.report import create_final_report

DEFAULT_JSON_OUTPUT: str = "results.json"


def init_pipeline(
    model_recto: str = DEFAULT_MODEL_RECTO,
    model_verso: str = DEFAULT_MODEL_VERSO,
) -> None:
    t0 = time.perf_counter()
    if not yolo_ready("recto"):
        load_model(model_recto, side="recto")
    if not yolo_ready("verso"):
        load_model(model_verso, side="verso")
    if not ocr_ready():
        load_engine()
    logger.info(f"Pipeline KYC initialisé (recto + verso) en {time.perf_counter() - t0:.2f}s.")


def run_pipeline(
    image_bytes: bytes,
    cin_type: str = "recto",
    json_output_path: str = DEFAULT_JSON_OUTPUT,
    output_image_path: str | None = None,
    save_crops_dir: str | None = "detectedimages",
) -> list[dict]:
    """
    Exécute le pipeline complet YOLO → OCR sur une image de CIN tunisienne.

    Args:
        image_bytes:       Image brute (JPEG/PNG) en bytes.
        cin_type:          "recto" ou "verso".
        json_output_path:  Chemin du fichier results.json de sortie.
        output_image_path: Chemin de l'image annotée (None = pas de sauvegarde).
        save_crops_dir:    Dossier pour sauvegarder les crops (None = pas de sauvegarde).

    Returns:
        Liste JSON, une entrée par détection :
        [{
            "cin_type":   "recto" | "verso",
            "class":      str,
            "yolo_score": float,
            "position":   {"x1": int, "y1": int, "x2": int, "y2": int},
            "crop_path":  str | None,
            "ocr_text":   str | None,
            "ocr_score":  float | None,
            "ocr_lines":  list[{"text": str, "score": float}] | None
        }, ...]
    """
    if cin_type not in ("recto", "verso"):
        raise ValueError(f"cin_type doit être 'recto' ou 'verso', reçu : {cin_type}")
    if not yolo_ready(cin_type):
        raise RuntimeError(f"YOLO [{cin_type}] non initialisé — appelez init_pipeline() d'abord.")
    if not ocr_ready():
        raise RuntimeError("PaddleOCR non initialisé — appelez init_pipeline() d'abord.")

    # ── 1. Détection YOLO + crops ─────────────────────────────────────────────
    t_start = time.perf_counter()
    detections = detect_and_crop(image_bytes, side=cin_type, save_dir=save_crops_dir)
    t_yolo = time.perf_counter()
    logger.info(f"YOLO [{cin_type}] : {t_yolo - t_start:.2f}s")
    if not detections:
        logger.warning(f"Aucune détection YOLO [{cin_type}] — résultats vides.")
        _save_json([], json_output_path)
        return []

    # ── 2. OCR par crop ───────────────────────────────────────────────────────
    json_output: list[dict] = []
    report_results: list[dict] = []
    # crop_bytes des champs skip_ocr (non sérialisables JSON) — stockés séparément
    skip_crop_bytes: dict[str, bytes] = {}

    for i, det in enumerate(detections):
        label = det["label"]
        pos   = det["position"]

        entry: dict = {
            "cin_type":   cin_type,
            "class":      label,
            "yolo_score": det["yolo_score"],
            "position":   pos,
            "crop_path":  det["crop_path"],
            "ocr_raw":    None,
            "ocr_text":   None,
            "ocr_score":  None,
            "ocr_lines":  None,
        }

        if det["skip_ocr"]:
            logger.debug(f"Skip OCR : {label}")
            skip_crop_bytes[label] = det["crop_bytes"]   # hors du dict sérialisé
            json_output.append(entry)
            continue

        ocr_out = run_ocr(
            crop_bytes=det["crop_bytes"],
            label=label,
            detection_index=i,
            position=pos,
        )

        entry["ocr_raw"]   = ocr_out.get("ocr_raw")
        entry["ocr_text"]  = ocr_out["ocr_text"]
        entry["ocr_score"] = ocr_out["ocr_score"]
        entry["ocr_lines"] = ocr_out["ocr_lines"]

        if ocr_out["ocr_text"] is not None:
            report_results.append({
                "class": label,
                "text":  ocr_out["ocr_text"],
                "score": ocr_out["ocr_score"] or 0.0,
            })

        json_output.append(entry)

    # ── 3. Annotation visuelle ────────────────────────────────────────────────
    if output_image_path:
        nparr = np.frombuffer(image_bytes, np.uint8)
        orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if orig_img is not None:
            _save_annotated(orig_img, detections, output_image_path)

    # ── 4. JSON final ─────────────────────────────────────────────────────────
    t_end = time.perf_counter()
    yolo_ms = int((t_yolo - t_start) * 1000)
    ocr_ms  = int((t_end  - t_yolo)  * 1000)
    logger.info(f"OCR [{cin_type}] : {ocr_ms}ms | YOLO : {yolo_ms}ms | Total : {yolo_ms + ocr_ms}ms")

    _save_json(json_output, json_output_path)

    # Réinjecte crop_bytes dans les entrées après sérialisation JSON (non stocké sur disque)
    for entry in json_output:
        if entry["class"] in skip_crop_bytes:
            entry["crop_bytes"] = skip_crop_bytes[entry["class"]]

    return json_output, yolo_ms, ocr_ms


def init_pipeline_passport(model_passport: str = DEFAULT_MODEL_PASSPORT) -> None:
    t0 = time.perf_counter()
    if not is_loaded_passport():
        load_model(model_passport, side="passport")
    if not ocr_ready():
        load_engine()
    logger.info(f"Pipeline passeport initialisé en {time.perf_counter() - t0:.2f}s.")


def run_pipeline_passport(
    image_bytes: bytes,
    json_output_path: str = DEFAULT_JSON_OUTPUT,
    output_image_path: str | None = None,
    save_crops_dir: str | None = "detectedimages",
) -> list[dict]:
    """
    Pipeline complet YOLO → OCR sur une image de passeport tunisien.
    Retourne la même structure que run_pipeline() avec doc_type='passport'.
    """
    if not is_loaded_passport():
        raise RuntimeError("YOLO [passport] non initialisé — appelez init_pipeline_passport() d'abord.")
    if not ocr_ready():
        raise RuntimeError("PaddleOCR non initialisé — appelez init_pipeline_passport() d'abord.")

    t_start = time.perf_counter()
    detections = detect_and_crop(image_bytes, side="passport", save_dir=save_crops_dir)
    t_yolo = time.perf_counter()
    logger.info(f"YOLO [passport] : {t_yolo - t_start:.2f}s")

    if not detections:
        logger.warning("Aucune détection YOLO [passport] — résultats vides.")
        _save_json([], json_output_path)
        return []

    json_output: list[dict] = []
    skip_crop_bytes: dict[str, bytes] = {}

    for i, det in enumerate(detections):
        label = det["label"]
        entry: dict = {
            "cin_type":   "passport",
            "class":      label,
            "yolo_score": det["yolo_score"],
            "position":   det["position"],
            "crop_path":  det["crop_path"],
            "ocr_raw":    None,
            "ocr_text":   None,
            "ocr_score":  None,
            "ocr_lines":  None,
        }

        if det["skip_ocr"]:
            logger.debug(f"Skip OCR : {label}")
            skip_crop_bytes[label] = det["crop_bytes"]
            json_output.append(entry)
            continue

        ocr_out = run_ocr(
            crop_bytes=det["crop_bytes"],
            label=label,
            detection_index=i,
            position=det["position"],
            doc_type="passport",
        )
        entry["ocr_raw"]   = ocr_out.get("ocr_raw")
        entry["ocr_text"]  = ocr_out["ocr_text"]
        entry["ocr_score"] = ocr_out["ocr_score"]
        entry["ocr_lines"] = ocr_out["ocr_lines"]
        json_output.append(entry)

    if output_image_path:
        nparr = np.frombuffer(image_bytes, np.uint8)
        orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if orig_img is not None:
            _save_annotated(orig_img, detections, output_image_path)

    t_end = time.perf_counter()
    yolo_ms = int((t_yolo - t_start) * 1000)
    ocr_ms  = int((t_end  - t_yolo)  * 1000)
    logger.info(f"OCR [passport] : {ocr_ms}ms | YOLO : {yolo_ms}ms | Total : {yolo_ms + ocr_ms}ms")
    _save_json(json_output, json_output_path)

    for entry in json_output:
        if entry["class"] in skip_crop_bytes:
            entry["crop_bytes"] = skip_crop_bytes[entry["class"]]

    return json_output, yolo_ms, ocr_ms


def run_pipeline_from_path(
    image_path: str,
    cin_type: str = "recto",
    json_output_path: str = DEFAULT_JSON_OUTPUT,
    output_image_path: str | None = None,
    save_crops_dir: str | None = "detectedimages",
) -> list[dict]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return run_pipeline(image_bytes, cin_type, json_output_path, output_image_path, save_crops_dir)


def _save_json(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON sauvegardé : {path}")


def _save_annotated(img: np.ndarray, detections: list[dict], output_path: str) -> None:
    annotated = img.copy()
    for det in detections:
        p = det["position"]
        cv2.rectangle(annotated, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 255, 0), 2)
    cv2.imwrite(output_path, annotated)
    logger.info(f"Image annotée : {output_path}")


# ── Point d'entrée CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "cin.png"
    CIN_TYPE   = sys.argv[2] if len(sys.argv) > 2 else "recto"
    JSON_OUT   = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_JSON_OUTPUT
    IMG_OUT    = sys.argv[4] if len(sys.argv) > 4 else "result_annotated.png"

    init_pipeline()
    results = run_pipeline_from_path(IMAGE_PATH, CIN_TYPE, JSON_OUT, IMG_OUT)

    if results:
        print(f"Pipeline terminé [{CIN_TYPE}] — {len(results)} entrée(s) dans {JSON_OUT}")
        img_annotated = cv2.imread(IMG_OUT)
        report_data = [
            {"class": r["class"], "text": r["ocr_text"] or "", "score": r["ocr_score"] or 0.0}
            for r in results if r["ocr_text"]
        ]
        if report_data and img_annotated is not None:
            create_final_report(img_annotated, report_data, "RAPPORT_FINAL_PFE.png", cin_type=CIN_TYPE)
    else:
        print("Aucun résultat.")
