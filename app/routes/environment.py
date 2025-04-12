from fastapi import APIRouter, UploadFile, File
import shutil
from app.services.yamnet_predict import predict_tags

router = APIRouter()


@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_tags(file_path)
    return {"detected": result}
