import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image

# Загрузка модели CNN
model = tf.keras.models.load_model("app/models/cnn_classifier.h5", compile=False)

# Функция для создания мел-спектрограммы и сохранения её как изображение
def create_mel_spectrogram(file_path, save_path="temp.png", sr=16000, n_mels=128):
    audio, _ = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(2.56, 2.56))  # Размер изображения 256x256
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Функция предсказания
def predict_audio(file_path):
    try:
        # Создание мел-спектрограммы
        create_mel_spectrogram(file_path)

        # Загрузка мел-спектрограммы как изображения
        img = Image.open("temp.png").convert("RGB").resize((128, 128))  # Убираем альфа-канал
        img_array = np.array(img) / 255.0  # Нормализация
        img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для batch

        # Предсказание
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)

        classes = ["Ferrari", "Audi", "Unknown"]
        confidence = prediction[0][class_idx]  # Уверенность модели в предсказании

        return f"Prediction: {classes[class_idx]} (Confidence: {confidence:.2f})"
    except Exception as e:
        return f"Prediction failed: {e}"
