"""
main.py — classification-service (port 8002) — STUB
Retourne une classification fictive jusqu'à l'implémentation réelle de YOLOcls.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from routers.classify import router

app = FastAPI(title="classification-service — STUB", version="0.1.0")
app.include_router(router)


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "service": "classification-service", "port": 8002, "stub": True}
