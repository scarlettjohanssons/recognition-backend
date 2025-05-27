import os
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Шляхи
input_dir = '/Users/illiakrytskyi/Documents/KNU/Kursova/DATASETS/RECOGNITION3s'
output_dir = 'app/data/test_stft_spectrograms'  # інша папка для збереження

# Очистити стару папку
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"🧹 Папку {output_dir} успішно очищено!")
os.makedirs(output_dir, exist_ok=True)

# Налаштування
target_sr = 22050
dpi = 100
img_size = (4.5, 4.5)
total_count = 0

for label in os.listdir(input_dir):
    class_dir = os.path.join(input_dir, label)
    if not os.path.isdir(class_dir):
        continue

    output_class_dir = os.path.join(output_dir, label)
    os.makedirs(output_class_dir, exist_ok=True)

    for file in os.listdir(class_dir):
        if not file.lower().endswith('.wav'):
            continue

        file_path = os.path.join(class_dir, file)
        try:
            y, sr = librosa.load(file_path, sr=target_sr)
        except Exception as e:
            print(f"⛔️ Проблема з файлом {file_path}: {e}")
            continue

        # Перевірка сили сигналу
        if np.max(np.abs(y)) < 0.01:
            continue

        # STFT спектограма
        D = np.abs(librosa.stft(y))
        D_dB = librosa.amplitude_to_db(D, ref=np.max)
        D_dB = np.clip(D_dB + 40, a_min=-80, a_max=0)

        # Візуалізація
        fig = plt.figure(figsize=img_size, dpi=dpi)
        plt.axis('off')
        librosa.display.specshow(D_dB, sr=target_sr, cmap='magma', x_axis='time', y_axis='linear')
        plt.tight_layout()

        output_file = os.path.join(output_class_dir, file.replace('.wav', '.png'))
        fig.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        total_count += 1
        if total_count % 500 == 0:
            print(f"🔍 Оброблено {total_count} файлів...")

print(f"✅ Готово. Згенеровано {total_count} STFT-спектограм у {output_dir}")
