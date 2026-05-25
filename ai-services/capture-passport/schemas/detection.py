"""
schemas/detection.py
Modèles Pydantic pour la capture passeport (WebSocket + HTTP).
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict


class GuideBox(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class ScreenSize(BaseModel):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class DetectionRequest(BaseModel):
    frame: str
    guide: GuideBox
    screen: ScreenSize


class DetectionResponse(BaseModel):
    status: Literal["ABSENT", "DETECTED", "CONFIRMED"]
    confidence: Optional[float] = None
    is_inside_guide: bool = False
    should_capture: bool = False
    capture_id: Optional[str] = None
    image_url: Optional[str] = None
    debug: Optional[Dict[str, object]] = None


class CaptureRequest(BaseModel):
    frame: str
    capture_id: str


class CaptureResponse(BaseModel):
    success: bool
    capture_id: str
    image_path: str
