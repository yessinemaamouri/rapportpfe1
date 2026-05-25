from fastapi import APIRouter

router = APIRouter()


@router.post("/classify", tags=["Classification"])
async def classify_document(payload: dict):
    """STUB — retourne une classification fictive."""
    return {"type": "CIN_RECTO", "confidence": 0.95, "stub": True}
