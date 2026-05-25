"""
debug_pipeline.py
Script de diagnostic du pipeline OCR — sans aucune modification du code existant.
Exécute YOLO + PaddleOCR sur les 3 documents de test et génère une structure debug/ complète.

Usage :
    cd ai-services/ocr-service
    python debug_pipeline.py
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# ── Résolution des imports (script lancé depuis le dossier ocr-service) ────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from services.service_yolo import load_model, detect_and_crop
from services.service_ocr import load_engine, run_ocr

# ── Constantes ─────────────────────────────────────────────────────────────────
DOCS_DIR  = ROOT / "documentsAtester"
DEBUG_DIR = ROOT / "debug"

DOCUMENTS = {
    "cin_recto":  {"file": DOCS_DIR / "cinRecto.jpg",  "side": "recto"},
    "cin_verso":  {"file": DOCS_DIR / "cinVerso.jpg",  "side": "verso"},
    "passport":   {"file": DOCS_DIR / "passPort.jpg",  "side": "passport"},
}

MODEL_PATHS = {
    "recto":    "models/modelYolo_11_fit_recto/best.pt",
    "verso":    "models/modeleYolo11_verso_fit/best.pt",
    "passport": "models/modelYolo_passport/best.pt",
}


# ── Helpers ─────────────────────────────────────────────────────────────────────

def make_dirs(doc_key: str) -> dict[str, Path]:
    base = DEBUG_DIR / doc_key
    dirs = {
        "base":         base,
        "crops":        base / "crops",
        "preprocessed": base / "preprocessed",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_models_once(sides: list[str]) -> None:
    print("\n=== Chargement des modèles YOLO ===")
    for side in sides:
        path = str(ROOT / MODEL_PATHS[side])
        print(f"  [{side}] chargement depuis {path} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        load_model(path, side=side)
        print(f"OK ({time.perf_counter() - t0:.1f}s)")

    print("\n=== Chargement PaddleOCR ===")
    t0 = time.perf_counter()
    load_engine()
    print(f"  PaddleOCR chargé ({time.perf_counter() - t0:.1f}s)")


def save_preprocessed_crop(crop_bytes: bytes, label: str, idx: int,
                            preprocessed_dir: Path) -> str:
    """
    Reproduit exactement le preprocessing de service_ocr._preprocess()
    et sauvegarde l'image envoyée à PaddleOCR.
    """
    nparr = np.frombuffer(crop_bytes, np.uint8)
    crop_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if crop_bgr is None:
        return ""

    h, w = crop_bgr.shape[:2]
    _MIN_TEXT_HEIGHT = 48
    _MAX_SIDE = 3800
    _FORCE_UPSCALE_LABELS = {"profession", "mother_name"}
    _FORCE_UPSCALE_LABELS_PASSPORT = {
        "profession", "mother_name", "full_name", "address",
        "dob", "issue_date", "expiry_date", "num_pass", "num_cin",
    }

    img = crop_bgr.copy()
    if label in _FORCE_UPSCALE_LABELS_PASSPORT or label in _FORCE_UPSCALE_LABELS:
        img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)
    elif h < _MIN_TEXT_HEIGHT:
        scale = _MIN_TEXT_HEIGHT / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    h2, w2 = img.shape[:2]
    if max(h2, w2) > _MAX_SIDE:
        scale = _MAX_SIDE / max(h2, w2)
        img = cv2.resize(img, (int(w2 * scale), int(h2 * scale)), interpolation=cv2.INTER_AREA)

    out_path = preprocessed_dir / f"{label}_{idx}_preprocessed.png"
    cv2.imwrite(str(out_path), img)
    return str(out_path)


def run_document(doc_key: str, doc_cfg: dict, dirs: dict[str, Path]) -> dict:
    """
    Exécute le pipeline YOLO+OCR sur UN document et retourne le rapport complet.
    """
    img_path: Path = doc_cfg["file"]
    side: str      = doc_cfg["side"]
    doc_type       = "passport" if side == "passport" else "cin"

    print(f"\n{'='*60}")
    print(f"DOCUMENT : {doc_key}  |  fichier : {img_path.name}  |  side : {side}")
    print(f"{'='*60}")

    if not img_path.exists():
        msg = f"FICHIER INTROUVABLE : {img_path}"
        print(f"  ERREUR — {msg}")
        return {"error": msg, "detections": [], "ocr_results": []}

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    # ── Sauvegarde original ─────────────────────────────────────────────────────
    orig_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if orig_img is not None:
        cv2.imwrite(str(dirs["base"] / "original.jpg"), orig_img)
        img_h, img_w = orig_img.shape[:2]
        print(f"  Image originale : {img_w}×{img_h} px")
    else:
        img_h, img_w = 0, 0
        print("  ERREUR : impossible de décoder l'image originale")

    # ── Étape 1 : YOLO ──────────────────────────────────────────────────────────
    print(f"\n--- Étape 1 : Détection YOLO [{side}] ---")
    t_yolo = time.perf_counter()
    try:
        detections = detect_and_crop(image_bytes, side=side, save_dir=str(dirs["crops"]))
    except Exception as exc:
        print(f"  ERREUR YOLO : {exc}")
        traceback.print_exc()
        return {"error": str(exc), "detections": [], "ocr_results": []}
    elapsed_yolo = time.perf_counter() - t_yolo

    print(f"  {len(detections)} détection(s) en {elapsed_yolo:.2f}s")
    print()

    detection_summary = []
    for i, det in enumerate(detections):
        p    = det["position"]
        bw   = p["x2"] - p["x1"]
        bh   = p["y2"] - p["y1"]

        nparr = np.frombuffer(det["crop_bytes"], np.uint8)
        crop  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        crop_w, crop_h = (crop.shape[1], crop.shape[0]) if crop is not None else (0, 0)

        print(f"  [{i:02d}] label={det['label']:<20} conf={det['yolo_score']:.4f}"
              f"  bbox=({p['x1']},{p['y1']},{p['x2']},{p['y2']})"
              f"  bbox_wh={bw}×{bh}"
              f"  crop_wh={crop_w}×{crop_h}"
              f"  skip_ocr={det['skip_ocr']}")

        detection_summary.append({
            "index":      i,
            "label":      det["label"],
            "yolo_score": det["yolo_score"],
            "bbox":       {"x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"]},
            "bbox_size":  {"width": bw, "height": bh},
            "crop_size":  {"width": crop_w, "height": crop_h},
            "skip_ocr":   det["skip_ocr"],
            "crop_path":  det["crop_path"],
        })

    # ── Étape 2 : OCR par crop ──────────────────────────────────────────────────
    print(f"\n--- Étape 2 : OCR PaddleOCR sur {len(detections)} champ(s) ---")
    ocr_results = []
    t_ocr_total = 0.0

    for i, det in enumerate(detections):
        label = det["label"]

        if det["skip_ocr"]:
            print(f"  [{i:02d}] {label:<20} → SKIP OCR (visuel uniquement)")
            ocr_results.append({
                "index":     i,
                "label":     label,
                "skip_ocr":  True,
                "ocr_text":  None,
                "ocr_score": None,
                "ocr_lines": None,
                "lang_used": "none",
                "elapsed_s": 0.0,
                "preprocessed_path": None,
            })
            continue

        # Sauvegarde du crop tel qu'envoyé à PaddleOCR (après preprocessing)
        pre_path = save_preprocessed_crop(
            det["crop_bytes"], label, i, dirs["preprocessed"]
        )

        t0 = time.perf_counter()
        try:
            ocr_out = run_ocr(
                crop_bytes=det["crop_bytes"],
                label=label,
                detection_index=i,
                position=det["position"],
                doc_type=doc_type,
            )
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  [{i:02d}] {label:<20} → ERREUR OCR : {exc}")
            traceback.print_exc()
            ocr_out = {"ocr_text": None, "ocr_score": None, "ocr_lines": None}

        t_ocr_total += elapsed

        # Détermine la langue utilisée (logique identique à service_ocr.py)
        _DIGITS_ONLY   = {"num_cin", "print_id"}
        _MIXED_CIN     = {"dob", "issue_date"}
        _DIGITS_PASS   = {"num_pass", "num_cin"}
        _MIXED_PASS    = {"dob", "issue_date", "expiry_date"}
        _LATIN_PASS    = {"last_name", "first_name", "pob"}

        if doc_type == "passport":
            if label in _LATIN_PASS:
                lang = "fr (latin)"
            elif label in _DIGITS_PASS:
                lang = "fr (chiffres)"
            elif label in _MIXED_PASS:
                lang = "fr (dates latin)"
            else:
                lang = "ar"
        else:
            if label in _DIGITS_ONLY:
                lang = "fr (chiffres)"
            elif label in _MIXED_CIN:
                lang = "ar + fr (date mixte)"
            else:
                lang = "ar"

        status = "OK" if ocr_out["ocr_text"] else "VIDE"
        score_str = f"{ocr_out['ocr_score']:.4f}" if ocr_out["ocr_score"] is not None else "N/A"
        print(f"  [{i:02d}] {label:<20} [{lang:<18}] → {status}"
              f"  score={score_str}"
              f"  texte={repr(ocr_out['ocr_text'])}"
              f"  ({elapsed:.2f}s)")

        ocr_results.append({
            "index":              i,
            "label":              label,
            "skip_ocr":           False,
            "lang_used":          lang,
            "ocr_text":           ocr_out["ocr_text"],
            "ocr_score":          ocr_out["ocr_score"],
            "ocr_lines":          ocr_out["ocr_lines"],
            "elapsed_s":          round(elapsed, 3),
            "preprocessed_path":  pre_path,
        })

    print(f"\n  Temps total OCR : {t_ocr_total:.2f}s")

    # ── Sauvegarde results.json ─────────────────────────────────────────────────
    results = {
        "document":        doc_key,
        "file":            str(img_path),
        "side":            side,
        "image_size":      {"width": img_w, "height": img_h},
        "yolo_elapsed_s":  round(elapsed_yolo, 3),
        "ocr_elapsed_s":   round(t_ocr_total, 3),
        "total_detections": len(detections),
        "total_ocr_ok":    sum(1 for r in ocr_results if r["ocr_text"] is not None),
        "total_ocr_empty": sum(1 for r in ocr_results if not r["skip_ocr"] and r["ocr_text"] is None),
        "total_skipped":   sum(1 for r in ocr_results if r["skip_ocr"]),
        "detections":      detection_summary,
        "ocr_results":     ocr_results,
    }

    json_path = dirs["base"] / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  results.json → {json_path}")

    return results


def print_analysis(all_results: dict) -> None:
    """Analyse finale consolidée pour identifier la source des problèmes."""
    print("\n" + "="*60)
    print("ANALYSE DIAGNOSTIQUE FINALE")
    print("="*60)

    for doc_key, res in all_results.items():
        if "error" in res and not res.get("detections"):
            print(f"\n[{doc_key}] ERREUR : {res['error']}")
            continue

        detections  = res.get("detections", [])
        ocr_results = res.get("ocr_results", [])
        total       = res.get("total_detections", 0)
        ok          = res.get("total_ocr_ok", 0)
        empty       = res.get("total_ocr_empty", 0)
        skipped     = res.get("total_skipped", 0)

        print(f"\n[{doc_key}]")
        print(f"  Détections YOLO : {total}")
        print(f"  OCR réussi      : {ok}")
        print(f"  OCR vide        : {empty}")
        print(f"  Skip (visuels)  : {skipped}")

        # Champs avec OCR vide
        failed_fields = [r for r in ocr_results if not r["skip_ocr"] and r["ocr_text"] is None]
        if failed_fields:
            print(f"\n  Champs OCR vides ({len(failed_fields)}) :")
            for r in failed_fields:
                det = next((d for d in detections if d["index"] == r["index"]), {})
                crop = det.get("crop_size", {})
                print(f"    - {r['label']:<20} crop={crop.get('width',0)}×{crop.get('height',0)}"
                      f"  lang={r['lang_used']}")
        else:
            print("  Tous les champs OCR ont retourné du texte.")

        # Crops trop petits
        small_crops = [
            d for d in detections
            if not d["skip_ocr"] and (d["crop_size"]["width"] < 80 or d["crop_size"]["height"] < 30)
        ]
        if small_crops:
            print(f"\n  Crops suspects (trop petits) ({len(small_crops)}) :")
            for d in small_crops:
                print(f"    - {d['label']:<20} crop={d['crop_size']['width']}×{d['crop_size']['height']}")

        # Faible confiance YOLO
        low_conf = [d for d in detections if d["yolo_score"] < 0.4]
        if low_conf:
            print(f"\n  Détections faible confiance YOLO (<0.4) ({len(low_conf)}) :")
            for d in low_conf:
                print(f"    - {d['label']:<20} conf={d['yolo_score']:.4f}")

        # Faible score OCR (mais texte détecté)
        low_ocr = [r for r in ocr_results if not r["skip_ocr"] and r["ocr_text"]
                   and r["ocr_score"] is not None and r["ocr_score"] < 0.6]
        if low_ocr:
            print(f"\n  Score OCR faible (<0.6) ({len(low_ocr)}) :")
            for r in low_ocr:
                print(f"    - {r['label']:<20} score={r['ocr_score']:.4f}  texte={repr(r['ocr_text'][:40])}")

    print("\n" + "="*60)
    print("Structure debug générée :")
    for doc_key in all_results:
        base = DEBUG_DIR / doc_key
        print(f"  {base}/")
        print(f"    original.jpg")
        print(f"    results.json")
        print(f"    crops/          ← crops YOLO bruts")
        print(f"    preprocessed/   ← images envoyées à PaddleOCR")
    print("="*60)


# ── Point d'entrée ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(ROOT)  # s'assurer que les chemins relatifs (models/) fonctionnent

    print("="*60)
    print("DEBUG PIPELINE OCR — Diagnostic sans modification")
    print("="*60)

    # Vérifie les documents
    print("\nVérification des documents de test :")
    missing = []
    for doc_key, cfg in DOCUMENTS.items():
        exists = cfg["file"].exists()
        status = "OK" if exists else "MANQUANT"
        print(f"  {cfg['file'].name:<20} → {status}")
        if not exists:
            missing.append(doc_key)

    if missing:
        print(f"\nATTENTION : {len(missing)} document(s) manquant(s) : {missing}")
        print("Continuer avec les documents disponibles...")

    # Charge tous les modèles une seule fois
    sides_needed = list({cfg["side"] for k, cfg in DOCUMENTS.items() if k not in missing})
    try:
        load_models_once(sides_needed)
    except Exception as exc:
        print(f"\nERREUR chargement modèles : {exc}")
        traceback.print_exc()
        sys.exit(1)

    # Traite chaque document
    all_results = {}
    for doc_key, doc_cfg in DOCUMENTS.items():
        if doc_key in missing:
            all_results[doc_key] = {"error": "fichier manquant", "detections": [], "ocr_results": []}
            continue
        dirs = make_dirs(doc_key)
        all_results[doc_key] = run_document(doc_key, doc_cfg, dirs)

    # Analyse finale
    print_analysis(all_results)

    print("\nDiagnostic terminé.")
