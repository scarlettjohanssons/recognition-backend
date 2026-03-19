import os
import random
import shutil
import librosa
import soundfile as sf
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Конфіг ---
source_dir = "/Users/illiakrytskyi/Documents/KNU/Kursova/DATASETS/RECOGNITION_MULTI"
output_dir = "app/data/mixed_spectrograms"
label_csv = "app/data/mixed_labels.csv"
target_sr = 22050
num_mixes = 1000
min_classes = 2
max_classes = 3

# --- Очистка ---
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# --- Збір аудіо по класах ---
class_files = {
    cls: [os.path.join(source_dir, cls, f) for f in os.listdir(os.path.join(source_dir, cls)) if f.endswith('.wav')]
    for cls in os.listdir(source_dir)
    if os.path.isdir(os.path.join(source_dir, cls))
}

all_classes = list(class_files.keys())
records = []
counter = 0

# --- Генерація міксів ---
for i in range(num_mixes):
    chosen_classes = random.sample(all_classes, random.randint(min_classes, max_classes))
    signals = []
    min_len = float('inf')

    for cls in chosen_classes:
        file_path = random.choice(class_files[cls])
        y, _ = librosa.load(file_path, sr=target_sr)
        y = librosa.util.fix_length(y, size=target_sr * 5)  # 5 секунд
        signals.append(y)

    # Накладення
    combined = np.sum(signals, axis=0)
    combined /= np.max(np.abs(combined) + 1e-9)  # нормалізація

    # Спектограма
    S = librosa.feature.melspectrogram(y=combined, sr=target_sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = np.clip(S_dB + 40, -80, 0)

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=target_sr, cmap='inferno')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"mix_{counter}.png")
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    records.append({
        "filepath": out_path,
        "labels": ",".join(sorted(chosen_classes))
    })
    counter += 1

# --- Збереження CSV ---
df = pd.DataFrame(records)
df.to_csv(label_csv, index=False)
print(f"✅ Створено {counter} міксів. CSV: {label_csv}")
