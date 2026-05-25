"""
schemas/detection.py
Définit les modèles de données Pydantic pour l'API et le WebSocket.
Permet de standardiser et valider les échanges entre le front (Next.js) et le back.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict

class GuideBox(BaseModel):
    """
    Cadre guide affiché côté frontend (en pixels de la frame envoyée).
    Le backend vérifie que la bbox YOLO de la CIN est contenue à l'intérieur de ce cadre.
    """
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class ScreenSize(BaseModel):
    """
    Dimensions écran réelles côté frontend (pixels viewport).
    """
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class DetectionRequest(BaseModel):
    """
    Payload reçu via WebSocket pour chaque frame.
    - frame: image base64 (data URL ou base64 brut)
    - guide: cadre guide en pixels (même repère que la frame envoyée)
    """
    frame: str
    guide: GuideBox
    screen: ScreenSize


class DetectionResponse(BaseModel):
    """
    Modèle de réponse envoyé en continu via WebSocket.
    Le frontend colore le cadre en fonction de `status`.
    """
    status: Literal["ABSENT", "DETECTED", "CONFIRMED"]
    confidence: Optional[float] = None
    is_inside_guide: bool = False
    should_capture: bool = False
    capture_id: Optional[str] = None
    image_url: Optional[str] = None
    debug: Optional[Dict[str, object]] = None

class CaptureRequest(BaseModel):
    """
    Modèle de requête reçu sur POST /capture lorsque le bouton est cliqué côté frontend.
    """
    frame: str       # Image capturée encodée en base64
    capture_id: str  # ID unique de la session de capture ou de validation

class CaptureResponse(BaseModel):
    """
    Modèle de réponse renvoyé au frontend une fois l'image traitée (croppée) et sauvegardée.
    """
    success: bool
    capture_id: str
    image_path: str
