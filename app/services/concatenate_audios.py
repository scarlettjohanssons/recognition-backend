import librosa
import soundfile as sf
import numpy as np


def concatenate_audios(input_paths, output_path, target_sr=22050):
    audios = []

    for path in input_paths:
        y, sr = librosa.load(path, sr=target_sr)
        audios.append(y)

    # Вирівняти довжину по мінімальному або максимальному значенню
    min_len = min(len(y) for y in audios)
    trimmed = [y[:min_len] for y in audios]

    # Накладення: елементна сума сигналів
    overlaid = np.sum(trimmed, axis=0)

    # Нормалізація до [-1, 1] щоб не було кліпінгу
    overlaid /= np.max(np.abs(overlaid) + 1e-9)

    sf.write(output_path, overlaid, samplerate=target_sr)
    print(f"✅ Накладене аудіо збережено: {output_path}")


# 👇 Автоматичний виклик при запуску скрипта
if __name__ == "__main__":
    input_files = [
        "/Users/illiakrytskyi/Documents/KNU/Kursova/DATASETS/TEST/dog_1_part_3.wav",
        "/Users/illiakrytskyi/Documents/KNU/Kursova/DATASETS/TEST/wind_1_part_86.wav"
    ]
    output_file = "/Users/illiakrytskyi/Documents/KNU/Kursova/DATASETS/TEST/dog_wind.wav"

    concatenate_audios(input_files, output_file)
