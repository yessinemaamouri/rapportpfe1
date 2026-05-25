from pydantic import BaseModel
from typing import Literal, Optional


class ClassifyRequest(BaseModel):
    minio_key: str
    document_id: str
    declared_type: str


class ClassifyResponse(BaseModel):
    document_id: str
    detected_type: Literal["CIN_RECTO", "CIN_VERSO", "PASSPORT", "ATTESTATION_TRAVAIL", "FICHE_PAIE", "UNKNOWN"]
    confidence: float
    match_client: bool
    flag: Optional[Literal["CLASSIFICATION_MISMATCH"]] = None
