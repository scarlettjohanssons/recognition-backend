import tensorflow_hub as hub
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd

# Завантаження моделі YAMNet
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Завантаження назв класів
class_map_path = "app/data/yamnet_class_map.csv"

def load_class_names():
    df = pd.read_csv(class_map_path)
    return df['display_name'].to_list()

class_names = load_class_names()

# Прогнозування тегів
def predict_tags(file_path, top_k=5, threshold=0.1):
    waveform, sr = librosa.load(file_path, sr=16000)
    scores, embeddings, spectrogram = yamnet_model(waveform)

    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_indices = mean_scores.argsort()[::-1][:top_k]

    results = []
    for i in top_indices:
        if mean_scores[i] > threshold:
            results.append(class_names[i])
    return results
