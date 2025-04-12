# app/main.py

from fastapi import FastAPI
from app.routes.audio import router as audio_bp
from app.routes.environment import router as environment_router

app = FastAPI(title="Sound Recognition API")

app.include_router(audio_bp, prefix="/audio_bp")
app.include_router(environment_router, prefix="/environment")


@app.get("/")
def root():
    return {"message": "FastAPI is working 🚀"}

