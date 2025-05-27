import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from fastapi import APIRouter, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

router = APIRouter()

# Завантажити модель один раз
model = load_model('app/models/best_model.keras')

# Класи (в тому ж порядку, що і в train_generator.class_indices)
class_labels = sorted(['Bike', 'Cat', 'Crow', 'Crowd', 'Dog', 'Elephant', 'Horse', 'Lion',
                       'Office', 'Parrot', 'Rainfall', 'Sparrow', 'Traffic', 'Train', 'Wind'])

@router.post("/predict")
async def predict_sound(file: UploadFile = File(...)):
    # Читання аудіо
    contents = await file.read()
    y, sr = librosa.load(io.BytesIO(contents), sr=22050)

    # Перевірка сигналу
    if np.max(np.abs(y)) < 0.01:
        return {"error": "Audio signal too weak or silent."}

    # Побудова спектограми (224×224 як під час train)
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
    preds = model.predict(img_array)
    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "class": class_labels[pred_idx],
        "confidence": round(confidence, 4)
    }
