from typing import Optional

from pydantic import BaseModel, Field


class FaceMatchRequest(BaseModel):
    cin_image_path: str = Field(..., description="Path to the CIN face image or YOLO crop.")
    selfie_image_path: str = Field(..., description="Path to the selfie image.")
    tolerance: float = Field(default=0.6, ge=0.0, le=2.0)


class FaceMatchResponse(BaseModel):
    match: bool
    similarity_score: float
    distance: Optional[float]
    status: str
    message: str
