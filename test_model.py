from app.services.recognition import predict_audio

# Путь к вашему новому аудио
audio_file = "app/data/ferrari_wav/ferrari-50.wav"

# Предсказание
result = predict_audio(audio_file)
print(f"Prediction result: {result}")
