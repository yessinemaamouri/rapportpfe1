import io
import os
import urllib3
from minio import Minio
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


class MinioService:
    def __init__(self):
        endpoint   = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        port       = os.getenv("MINIO_PORT", "9000")
        self._bucket = os.getenv("MINIO_BUCKET", "kyc-temp")
        use_ssl    = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")

        self._host     = endpoint if ":" in endpoint else f"{endpoint}:{port}"
        self._base_url = f"http{'s' if use_ssl else ''}://{self._host}"
        self._client   = Minio(
            self._host,
            access_key=access_key,
            secret_key=secret_key,
            secure=use_ssl,
            http_client=urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=2.0, read=2.0),
                retries=False,
            ),
        )
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def upload_selfie(self, image_bytes: bytes, capture_id: str) -> str:
        object_name = f"selfies/{capture_id}.jpg"
        self._client.put_object(
            bucket_name  = self._bucket,
            object_name  = object_name,
            data         = io.BytesIO(image_bytes),
            length       = len(image_bytes),
            content_type = "image/jpeg",
        )
        return f"{self._base_url}/{self._bucket}/{object_name}"
