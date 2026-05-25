import os
import psycopg2
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


class DbService:
    def __init__(self):
        self._database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:987123@localhost:5432/photodb",
        )

    def _connect(self):
        return psycopg2.connect(self._database_url, connect_timeout=3)

    def save_selfie(
        self,
        capture_id:    str,
        minio_url:     str,
        confidence:    float | None,
        face_detected: bool,
        user_id:       str | None = None,
        demande_id:    str | None = None,
    ) -> None:
        """
        Insère ou met à jour le selfie dans la table selfies.
        1 demande = 1 selfie (UNIQUE sur demande_id) → ON CONFLICT UPDATE.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO selfies
                        (demande_id, user_id, minio_path, confidence, face_detected, face_match_score, face_match_passe)
                    VALUES (%s, %s, %s, %s, %s, NULL, FALSE)
                    ON CONFLICT (demande_id) DO UPDATE
                        SET minio_path     = EXCLUDED.minio_path,
                            confidence     = EXCLUDED.confidence,
                            face_detected  = EXCLUDED.face_detected,
                            user_id        = EXCLUDED.user_id;
                """, (demande_id, user_id, minio_url, confidence, face_detected))
            conn.commit()
