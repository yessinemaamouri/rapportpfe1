"""
services/minio_client.py
Téléchargement d'images depuis MinIO pour le pipeline OCR.
"""
import io
import os
import urllib.parse

import urllib3
from dotenv import load_dotenv
from minio import Minio

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


def _build_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    port = os.getenv("MINIO_PORT", "9000")
    use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
    host = endpoint if ":" in endpoint else f"{endpoint}:{port}"
    return Minio(
        host,
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=use_ssl,
        http_client=urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=5.0, read=10.0),
            retries=False,
        ),
    )


_client: Minio | None = None


def get_client() -> Minio:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def upload_bytes(data: bytes, object_name: str, content_type: str = "image/png") -> str:
    """
    Upload raw bytes to MinIO and return the internal HTTP URL.
    object_name: e.g. "kyc-temp/<demande_id>/face_crop.png"
    """
    import io as _io
    bucket = os.getenv("MINIO_BUCKET", "kyc-temp")
    client = get_client()
    client.put_object(
        bucket,
        object_name,
        _io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    port     = os.getenv("MINIO_PORT", "9000")
    host     = endpoint if ":" in endpoint else f"{endpoint}:{port}"
    return f"http://{host}/{bucket}/{object_name}"


def download_image(minio_url: str) -> bytes:
    """
    Télécharge une image depuis son URL MinIO complète.
    Ex: http://localhost:9000/kyc-temp/cin/session_recto.jpg
    """
    bucket = os.getenv("MINIO_BUCKET", "kyc-temp")
    parsed = urllib.parse.urlparse(minio_url)
    # Le path est /bucket/object_name → on retire le /bucket/
    object_name = parsed.path.lstrip("/")
    if object_name.startswith(bucket + "/"):
        object_name = object_name[len(bucket) + 1:]

    client = get_client()
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()
