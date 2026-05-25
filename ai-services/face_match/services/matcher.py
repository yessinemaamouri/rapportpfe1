"""
Face matching utilities based on face-recognition/dlib.

The public entry point is `face_match(cin_image_path, selfie_image_path)`.
It returns a JSON-serializable dictionary suitable for API responses and
backend workers.
"""
from __future__ import annotations

import math as _math
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import face_recognition
import numpy as np
from PIL import UnidentifiedImageError
from loguru import logger

DEFAULT_TOLERANCE = float(
    os.getenv("FACE_MATCH_TOLERANCE", os.getenv("FACE_MATCH_THRESHOLD", "0.50"))
)
NO_FACE_DETECTED = "NO_FACE_DETECTED"
MATCH_PROCESSED = "MATCH_PROCESSED"
INVALID_IMAGE = "INVALID_IMAGE"
FILE_NOT_FOUND = "FILE_NOT_FOUND"
PROCESSING_ERROR = "PROCESSING_ERROR"


def _resolve_image(src: str | Path, label: str) -> Path:
    """Return a local Path for *src*, downloading it first if it is an HTTP URL."""
    s = str(src)
    if s.startswith("http://") or s.startswith("https://"):
        suffix = Path(s.split("?")[0]).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            urllib.request.urlretrieve(s, tmp.name)
        except Exception as exc:
            raise FileNotFoundError(f"{label}: could not download URL: {exc}") from exc
        return Path(tmp.name)
    return Path(src)


def face_match(
    cin_image_path: str | Path,
    selfie_image_path: str | Path,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """
    Compare the face from a CIN image/crop with a selfie image.

    Args:
        cin_image_path: Local path to the CIN image or YOLO face crop.
        selfie_image_path: Local path to the selfie image.
        tolerance: Maximum euclidean distance accepted as a match.

    Returns:
        {
            "match": bool,
            "similarity_score": float,
            "distance": float | None,
            "status": str,
            "message": str,
        }
    """
    try:
        cin_encoding = _extract_primary_face_encoding(cin_image_path, "cin_image")
        selfie_encoding = _extract_primary_face_encoding(selfie_image_path, "selfie_image")

        distance = float(face_recognition.face_distance([cin_encoding], selfie_encoding)[0])
        similarity_score = _distance_to_similarity_score(distance, threshold=tolerance)
        is_match = distance <= tolerance

        return {
            "match": is_match,
            "similarity_score": similarity_score,
            "distance": round(distance, 6),
            "status": MATCH_PROCESSED,
            "message": "Face comparison completed successfully.",
        }

    except NoFaceDetectedError as exc:
        logger.warning(str(exc))
        return _error_result(NO_FACE_DETECTED, str(exc))
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        return _error_result(FILE_NOT_FOUND, str(exc))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning(f"Invalid face image input: {exc}")
        return _error_result(INVALID_IMAGE, "Image file is missing, corrupted, or unsupported.")
    except Exception as exc:
        logger.exception(f"Unexpected face matching error: {exc}")
        return _error_result(PROCESSING_ERROR, "Unexpected error while processing face match.")


class NoFaceDetectedError(IndexError):
    """Raised when face-recognition cannot extract an encoding from an image."""


def _extract_primary_face_encoding(image_path: str | Path, label: str) -> np.ndarray:
    """
    Load an image and return the encoding for its largest detected face.

    `face_locations` is computed once and passed to `face_encodings` to avoid
    doing duplicate detection work. HTTP(S) URLs are downloaded to a temp file.
    """
    path = _resolve_image(image_path, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: file not found: {path}")

    image = face_recognition.load_image_file(str(path))
    face_locations = face_recognition.face_locations(image, model="hog")
    if not face_locations:
        raise NoFaceDetectedError(f"{label}: no face detected.")

    primary_location = max(face_locations, key=_face_area)
    encodings = face_recognition.face_encodings(
        image,
        known_face_locations=[primary_location],
        num_jitters=1,
        model="small",
    )
    if not encodings:
        raise NoFaceDetectedError(f"{label}: face detected but encoding failed.")

    return encodings[0]


def _face_area(location: tuple[int, int, int, int]) -> int:
    top, right, bottom, left = location
    return max(0, right - left) * max(0, bottom - top)


def _distance_to_similarity_score(distance: float, threshold: float = DEFAULT_TOLERANCE) -> float:
    """
    Convertit la distance euclidienne face_recognition en score de similarité [0, 100].

    La distance brute (dlib) n'est pas une similarité :
      - Même personne (selfie ↔ photo doc)  → distance typique 0.25–0.48
      - Personnes différentes               → distance typique 0.50–0.90+

    Formule sigmoïde centrée sur le threshold (point de décision) :
      k = steepness (pente) — 12 donne une courbe bien séparée autour du seuil
      score = sigmoid(-k × (distance - threshold)) × 100

    Résultat avec threshold=0.50, k=12 :
      distance 0.25 → ~95 %   (même personne, très proche)
      distance 0.35 → ~88 %   (même personne, typique)
      distance 0.45 → ~73 %   (même personne, acceptable)
      distance 0.50 → ~50 %   (seuil de décision)
      distance 0.60 → ~18 %   (personne différente)
      distance 0.75 → ~4 %    (clairement différent)
    """
    k = 12.0
    score = 1.0 / (1.0 + _math.exp(k * (distance - threshold)))
    return round(max(0.0, min(100.0, score * 100.0)), 2)


def _error_result(status: str, message: str) -> dict[str, Any]:
    return {
        "match": False,
        "similarity_score": 0.0,
        "distance": None,
        "status": status,
        "message": message,
    }
