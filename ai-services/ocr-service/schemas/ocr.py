from pydantic import BaseModel
from typing import Optional


class OcrRequest(BaseModel):
    minio_url_recto: str
    minio_url_verso: str
    document_id: str


class OcrLineResult(BaseModel):
    text: str
    score: float


class BBoxPosition(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class OcrDetection(BaseModel):
    cin_type:   str
    field:      str
    yolo_score: float
    position:   Optional[BBoxPosition] = None
    ocr_raw:    Optional[str] = None
    ocr_text:   Optional[str] = None
    ocr_score:  Optional[float] = None
    ocr_lines:  Optional[list[OcrLineResult]] = None


class OcrStructured(BaseModel):
    nom: Optional[str] = None
    prenom: Optional[str] = None
    nom_complet: Optional[str] = None
    cin_number: Optional[str] = None
    date_naissance: Optional[str] = None
    lieu_naissance: Optional[str] = None
    adresse: Optional[str] = None
    profession: Optional[str] = None
    nom_mere: Optional[str] = None
    print_id: Optional[str] = None
    issue_date: Optional[str] = None


class OcrResponse(BaseModel):
    document_id:     str
    structured:      OcrStructured
    structured_raw:  dict                  # brut OCR avant post-traitement (cohérence avec passeport)
    raw_detections:  list[OcrDetection]
    confidence:      float
    yolo_elapsed_ms: Optional[int] = None
    ocr_elapsed_ms:  Optional[int] = None
    face_crop_url:   Optional[str] = None


# ── Passeport ─────────────────────────────────────────────────────────────────

class PassportOcrRequest(BaseModel):
    minio_url: str
    document_id: str


class PassportOcrStructured(BaseModel):
    last_name:     Optional[str] = None
    first_name:    Optional[str] = None
    full_name_ar:  Optional[str] = None
    nom_ar:        Optional[str] = None   # extrait par post-traitement
    prenom_ar:     Optional[str] = None   # extrait par post-traitement
    num_pass:      Optional[str] = None
    num_cin:       Optional[str] = None
    dob:           Optional[str] = None
    pob:           Optional[str] = None
    issue_date:    Optional[str] = None
    expiry_date:   Optional[str] = None
    address_ar:    Optional[str] = None
    profession_ar: Optional[str] = None


class PassportOcrResponse(BaseModel):
    document_id:     str
    structured:      PassportOcrStructured
    structured_raw:  dict                   # brut OCR avant post-traitement
    raw_detections:  list[OcrDetection]
    confidence:      float
    yolo_elapsed_ms: Optional[int] = None
    ocr_elapsed_ms:  Optional[int] = None
    face_crop_url:   Optional[str] = None
