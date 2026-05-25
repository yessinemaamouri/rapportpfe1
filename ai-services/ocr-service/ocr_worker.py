"""
ocr_worker.py — Worker OCR isolé pour un seul slot recto+verso.
Appelé en subprocess par test_kyc_pipeline.py pour isoler la mémoire GPU.

Usage:
    python ocr_worker.py <recto_path> <verso_path> <pair_idx> <output_json>
"""
import sys
import os
import json
from pathlib import Path

# CWD doit être ai-services/ocr-service/ (les modèles sont relatifs à ce répertoire)
HERE = Path(__file__).parent
os.chdir(str(HERE))
sys.path.insert(0, str(HERE))

from services.pipeline import run_pipeline, init_pipeline

_RECTO_FIELD_MAP = {
    "last_name":  "nom",
    "first_name": "prenom",
    "full_name":  "nom_complet",
    "num_cin":    "cin_number",
    "dob":        "date_naissance",
    "pob":        "lieu_naissance",
}
_VERSO_FIELD_MAP = {
    "mother_name": "nom_mere",
    "profession":  "profession",
    "address":     "adresse",
    "print_id":    "print_id",
    "issue_date":  "issue_date",
}


def _build_structured(detections, cin_type):
    field_map = _RECTO_FIELD_MAP if cin_type == "recto" else _VERSO_FIELD_MAP
    result = {}
    for det in detections:
        label = det.get("class", "")
        text  = det.get("ocr_text")
        if text and label in field_map:
            result[field_map[label]] = text
    return result


def _avg_confidence(detections):
    scores = [d["ocr_score"] for d in detections if d.get("ocr_score") is not None]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def main():
    if len(sys.argv) < 5:
        print("Usage: python ocr_worker.py <recto> <verso> <idx> <output.json>", file=sys.stderr)
        sys.exit(1)

    recto_path  = Path(sys.argv[1])
    verso_path  = Path(sys.argv[2])
    pair_idx    = int(sys.argv[3])
    output_json = Path(sys.argv[4])

    init_pipeline()

    recto_bytes = recto_path.read_bytes()
    verso_bytes = verso_path.read_bytes()

    os.makedirs("tmp_ocr", exist_ok=True)
    recto_dets = run_pipeline(
        recto_bytes, cin_type="recto",
        json_output_path=f"tmp_ocr/recto_{pair_idx}.json",
        output_image_path=None,
        save_crops_dir=None,
    )
    verso_dets = run_pipeline(
        verso_bytes, cin_type="verso",
        json_output_path=f"tmp_ocr/verso_{pair_idx}.json",
        output_image_path=None,
        save_crops_dir=None,
    )

    result = {
        "recto_structured": _build_structured(recto_dets, "recto"),
        "verso_structured":  _build_structured(verso_dets, "verso"),
        "recto_detections":  recto_dets,
        "verso_detections":  verso_dets,
        "confidence":        _avg_confidence(recto_dets + verso_dets),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"OK: {output_json}")


if __name__ == "__main__":
    main()
