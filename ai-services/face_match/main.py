"""
face_match service for eKYC.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8005 --reload
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI

from routers.match import router

app = FastAPI(title="face_match - eKYC biometric comparison", version="0.1.0")
app.include_router(router)


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "service": "face_match", "port": 8005}
