from fastapi import APIRouter

from schemas.match import FaceMatchRequest, FaceMatchResponse
from services.matcher import face_match

router = APIRouter()


@router.post("/face-match", response_model=FaceMatchResponse, tags=["Face Match"])
async def compare_faces(payload: FaceMatchRequest):
    """Compare a CIN/document face image with a selfie image."""
    return face_match(
        cin_image_path=payload.cin_image_path,
        selfie_image_path=payload.selfie_image_path,
        tolerance=payload.tolerance,
    )
