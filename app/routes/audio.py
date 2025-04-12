# app/routes/audio.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.recognition import predict_audio
import os
import shutil

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    file_path = f"temp/{file.filename}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_audio(file_path)
        return JSONResponse(content={"result": result})

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
