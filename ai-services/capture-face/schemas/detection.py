from pydantic import BaseModel
from typing import Optional, Literal, Dict


class DetectionResponse(BaseModel):
    status: Literal["ABSENT", "DETECTED", "CONFIRMED"]
    confidence: Optional[float] = None
    should_capture: bool = False
    capture_id: Optional[str] = None
    image_url: Optional[str] = None


class DetectRequest(BaseModel):
    image: str
