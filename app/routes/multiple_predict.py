import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from fastapi import APIRouter, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import pickle

router = APIRouter()

# 1. Завантаження моделі і класів
model = load_model('app/models/multi_finetuned_model.keras')
with open('app/data/label_classes.pkl', 'rb') as f:
    class_labels = pickle.load(f)  # ['Bike', 'Cat', ..., 'Wind']

@router.post("/predict")
async def predict_sound(file: UploadFile = File(...)):
    contents = await file.read()
    y, sr = librosa.load(io.BytesIO(contents), sr=22050)

    if np.max(np.abs(y)) < 0.01:
        return {"error": "Audio signal too weak or silent."}

    # Побудова спектограми
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis('off')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = np.clip(S_dB + 40, -80, 0)
    librosa.display.specshow(S_dB, sr=sr, cmap='inferno')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Обробка зображення
    img = Image.open(buf).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Прогноз
    preds = model.predict(img_array)[0]  # [num_classes] з sigmoid
    print("Raw sigmoid output:", preds)

    threshold = 0.3
    results = [
        {"class": cls, "confidence": round(float(conf), 4)}
        for cls, conf in zip(class_labels, preds)
        if conf > threshold
    ]

    return {"predicted": results}
