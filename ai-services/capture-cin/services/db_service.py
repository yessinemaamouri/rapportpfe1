"""
services/db_service.py
Persistance des captures CIN dans PostgreSQL.
"""
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


class DbService:
    def __init__(self) -> None:
        self._database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:987123@localhost:5432/photodb",
        )

    def _connect(self):
        return psycopg2.connect(self._database_url, connect_timeout=3)

    def init_table(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS cin_captures (
                      id          SERIAL PRIMARY KEY,
                      capture_id  VARCHAR(128) UNIQUE NOT NULL,
                      minio_url   TEXT NOT NULL,
                      confidence  FLOAT,
                      created_at  TIMESTAMP DEFAULT NOW()
                    );
                """)
                # Agrandit la colonne si elle existe déjà en VARCHAR(36)
                cur.execute("""
                    ALTER TABLE cin_captures
                    ALTER COLUMN capture_id TYPE VARCHAR(128);
                """)
            conn.commit()

    def save_capture(self, capture_id: str, minio_url: str, confidence: float | None) -> None:
        query = """
        INSERT INTO cin_captures (capture_id, minio_url, confidence)
        VALUES (%s, %s, %s)
        ON CONFLICT (capture_id) DO NOTHING;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (capture_id, minio_url, confidence))
            conn.commit()
