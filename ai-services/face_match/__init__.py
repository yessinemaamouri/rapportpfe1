"""Face matching service package."""

from .services.matcher import DEFAULT_TOLERANCE, face_match

__all__ = ["DEFAULT_TOLERANCE", "face_match"]
