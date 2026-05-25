"""
services/minio_client.py
Lecture des images depuis MinIO (lecture seule pour classification-service).
"""
import io
import os
from minio import Minio
import urllib3
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


class MinioClient:
    def __init__(self) -> None:
        endpoint = os.getenv("MINIO_ENDPOINT", "minio")
        port = os.getenv("MINIO_PORT", "9000")
        use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
        self._bucket = os.getenv("MINIO_BUCKET", "kyc-temp")
        self._client = Minio(
            f"{endpoint}:{port}",
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=use_ssl,
            http_client=urllib3.PoolManager(timeout=urllib3.Timeout(connect=3.0, read=10.0)),
        )

    def get_object(self, key: str) -> bytes:
        response = self._client.get_object(self._bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
