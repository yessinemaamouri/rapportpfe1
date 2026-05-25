"""
routers/ocr.py
POST /extract — pipeline complet YOLO → PaddleOCR sur images MinIO.
"""
import threading
from fastapi import APIRouter, HTTPException
from loguru import logger

from schemas.ocr import (
    OcrRequest, OcrResponse, OcrStructured, OcrDetection, OcrLineResult, BBoxPosition,
    PassportOcrRequest, PassportOcrResponse, PassportOcrStructured,
)
from services.pipeline import run_pipeline, init_pipeline, run_pipeline_passport, init_pipeline_passport
from services.minio_client import download_image, upload_bytes
from services.post_ocr_passport import enrich_passport_structured

# Flags lazy init — mis a jour par le warm-up background ou au premier appel
_cin_initialized      = False
_passport_initialized = False


def _set_cin_initialized():
    global _cin_initialized
    _cin_initialized = True


def _set_passport_initialized():
    global _passport_initialized
    _passport_initialized = True


router = APIRouter()

# Serialise les appels predict() — PaddleOCR et YOLO ne sont pas thread-safe
_pipeline_lock = threading.Lock()

# Mapping label YOLO → champ structuré
_PASSPORT_FIELD_MAP = {
    "last_name":   "last_name",
    "first_name":  "first_name",
    "full_name":   "full_name_ar",
    "num_pass":    "num_pass",
    "num_cin":     "num_cin",
    "dob":         "dob",
    "pob":         "pob",
    "issue_date":  "issue_date",
    "expiry_date": "expiry_date",
    "address":     "address_ar",
    "profession":  "profession_ar",
}

_RECTO_FIELD_MAP = {
    "last_name":   "nom",
    "first_name":  "prenom",
    "full_name":   "nom_complet",
    "num_cin":     "cin_number",
    "dob":         "date_naissance",
    "pob":         "lieu_naissance",
}
_VERSO_FIELD_MAP = {
    "mother_name": "nom_mere",
    "profession":  "profession",
    "address":     "adresse",
    "print_id":    "print_id",
    "issue_date":  "issue_date",
}


def _build_structured(detections: list[dict], cin_type: str, field: str = "ocr_text") -> dict:
    field_map = _RECTO_FIELD_MAP if cin_type == "recto" else _VERSO_FIELD_MAP
    result = {}
    for det in detections:
        label = det["class"]
        text = det.get(field)
        if text and label in field_map:
            result[field_map[label]] = text
    return result


@router.post("/extract", response_model=OcrResponse, tags=["OCR"])
def extract_text(req: OcrRequest):
    global _cin_initialized
    logger.info(f"OCR request — document_id={req.document_id}")
    if not _cin_initialized:
        with _pipeline_lock:
            if not _cin_initialized:
                logger.info("Lazy init pipeline CIN…")
                init_pipeline()
                _cin_initialized = True

    # Télécharge recto + verso depuis MinIO
    try:
        recto_bytes = download_image(req.minio_url_recto)
        verso_bytes = download_image(req.minio_url_verso)
        logger.info(f"Images téléchargées — recto={len(recto_bytes)}B verso={len(verso_bytes)}B")
    except Exception as e:
        logger.error(f"Téléchargement MinIO échoué: {e}")
        raise HTTPException(status_code=502, detail=f"MinIO download error: {e}")

    # Pipeline YOLO → OCR (sérialisé — YOLO/PaddleOCR non thread-safe)
    try:
        with _pipeline_lock:
            recto_detections, yolo_ms_r, ocr_ms_r = run_pipeline(recto_bytes, cin_type="recto", json_output_path="/tmp/recto.json")
            verso_detections, yolo_ms_v, ocr_ms_v = run_pipeline(verso_bytes, cin_type="verso", json_output_path="/tmp/verso.json")
        yolo_elapsed_ms = yolo_ms_r + yolo_ms_v
        ocr_elapsed_ms  = ocr_ms_r  + ocr_ms_v
    except RuntimeError as e:
        logger.error(f"Pipeline non initialisé: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Brut OCR avant post-traitement
    structured_raw = {}
    structured_raw.update(_build_structured(recto_detections, "recto", field="ocr_raw"))
    structured_raw.update(_build_structured(verso_detections, "verso", field="ocr_raw"))

    # Structured post-traité
    structured_dict = {}
    structured_dict.update(_build_structured(recto_detections, "recto"))
    structured_dict.update(_build_structured(verso_detections, "verso"))
    structured = OcrStructured(**structured_dict)

    # Construit raw_detections
    all_detections: list[OcrDetection] = []
    for det in recto_detections + verso_detections:
        lines = [OcrLineResult(**l) for l in det["ocr_lines"]] if det.get("ocr_lines") else None
        pos   = BBoxPosition(**det["position"]) if det.get("position") else None
        all_detections.append(OcrDetection(
            cin_type=det["cin_type"],
            field=det["class"],
            yolo_score=det["yolo_score"],
            position=pos,
            ocr_raw=det.get("ocr_raw"),
            ocr_text=det.get("ocr_text"),
            ocr_score=det.get("ocr_score"),
            ocr_lines=lines,
        ))

    # Score de confiance global = moyenne des scores OCR non nuls
    scores = [d.ocr_score for d in all_detections if d.ocr_score is not None]
    confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Upload du crop "image" (visage recto CIN) vers MinIO
    face_crop_url: str | None = None
    image_det = next(
        (d for d in recto_detections if d.get("class") == "image" and d.get("crop_bytes")),
        None,
    )
    if image_det:
        try:
            obj_name = f"face_crop/{req.document_id}.png"
            face_crop_url = upload_bytes(image_det["crop_bytes"], obj_name)
            logger.info(f"Face crop uploadé → {face_crop_url}")
        except Exception as _e:
            logger.warning(f"Upload face crop échoué (non bloquant) : {_e}")

    logger.info(f"OCR terminé — {len(all_detections)} détections, confidence={confidence}, yolo={yolo_elapsed_ms}ms, ocr={ocr_elapsed_ms}ms")
    return OcrResponse(
        document_id=req.document_id,
        structured=structured,
        structured_raw=structured_raw,
        raw_detections=all_detections,
        confidence=confidence,
        yolo_elapsed_ms=yolo_elapsed_ms,
        ocr_elapsed_ms=ocr_elapsed_ms,
        face_crop_url=face_crop_url,
    )


@router.post("/extract-passport", response_model=PassportOcrResponse, tags=["OCR"])
def extract_passport(req: PassportOcrRequest):
    global _passport_initialized
    logger.info(f"OCR passeport — document_id={req.document_id}")

    try:
        image_bytes = download_image(req.minio_url)
        logger.info(f"Image passeport téléchargée — {len(image_bytes)}B")
    except Exception as e:
        logger.error(f"Téléchargement MinIO échoué: {e}")
        raise HTTPException(status_code=502, detail=f"MinIO download error: {e}")

    if not _passport_initialized:
        with _pipeline_lock:
            if not _passport_initialized:
                try:
                    logger.info("Lazy init pipeline passeport…")
                    init_pipeline_passport()
                    _passport_initialized = True
                except Exception as e:
                    logger.error(f"Init pipeline passeport échouée: {e}")
                    raise HTTPException(status_code=503, detail=str(e))

    try:
        with _pipeline_lock:
            detections, yolo_elapsed_ms, ocr_elapsed_ms = run_pipeline_passport(image_bytes, json_output_path="/tmp/passport.json")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur pipeline passeport: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Brut OCR avant post-traitement
    structured_raw: dict = {}
    for det in detections:
        label = det["class"]
        raw = det.get("ocr_raw")
        if raw and label in _PASSPORT_FIELD_MAP:
            structured_raw[_PASSPORT_FIELD_MAP[label]] = raw

    # Structured post-traité
    structured_dict: dict = {}
    for det in detections:
        label = det["class"]
        text  = det.get("ocr_text")
        if text and label in _PASSPORT_FIELD_MAP:
            structured_dict[_PASSPORT_FIELD_MAP[label]] = text

    # Post-traitement : nom_ar / prenom_ar + normalisation dates
    enrich_passport_structured(structured_dict)

    structured = PassportOcrStructured(**structured_dict)

    # raw_detections
    all_detections: list[OcrDetection] = []
    for det in detections:
        lines = [OcrLineResult(**l) for l in det["ocr_lines"]] if det.get("ocr_lines") else None
        pos   = BBoxPosition(**det["position"]) if det.get("position") else None
        all_detections.append(OcrDetection(
            cin_type=det["cin_type"],
            field=det["class"],
            yolo_score=det["yolo_score"],
            position=pos,
            ocr_raw=det.get("ocr_raw"),
            ocr_text=det.get("ocr_text"),
            ocr_score=det.get("ocr_score"),
            ocr_lines=lines,
        ))

    scores     = [d.ocr_score for d in all_detections if d.ocr_score is not None]
    confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Upload du crop "image" (photo passeport) vers MinIO
    face_crop_url: str | None = None
    image_det_p = next(
        (d for d in detections if d.get("class") == "image" and d.get("crop_bytes")),
        None,
    )
    if image_det_p:
        try:
            obj_name = f"face_crop/{req.document_id}.png"
            face_crop_url = upload_bytes(image_det_p["crop_bytes"], obj_name)
            logger.info(f"Face crop passeport uploadé → {face_crop_url}")
        except Exception as _e:
            logger.warning(f"Upload face crop passeport échoué (non bloquant) : {_e}")

    logger.info(f"OCR passeport terminé — {len(all_detections)} détections, confidence={confidence}, yolo={yolo_elapsed_ms}ms, ocr={ocr_elapsed_ms}ms")
    return PassportOcrResponse(
        document_id=req.document_id,
        structured=structured,
        structured_raw=structured_raw,
        raw_detections=all_detections,
        confidence=confidence,
        yolo_elapsed_ms=yolo_elapsed_ms,
        ocr_elapsed_ms=ocr_elapsed_ms,
        face_crop_url=face_crop_url,
    )
