import librosa
import numpy as np
import soundfile as sf
import os
import random

# Папка с оригинальными файлами и папка для сохранения
input_folder = "app/data/audi_wav"
output_folder = "app/data/audi_augmented"
os.makedirs(output_folder, exist_ok=True)


# Функция для добавления шума (дождь, ветер, городской шум)
def add_random_noise(data, noise_folder="app/data/noises_wav", sr=16000):
    noise_files = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith('.wav')]
    if not noise_files:
        print("No noise files found. Skipping noise augmentation.")
        return data
    noise_file = random.choice(noise_files)
    noise, _ = librosa.load(noise_file, sr=sr)
    noise = np.resize(noise, len(data))  # Подгоняем длину шума под аудио
    return data + 0.02 * noise  # Добавляем шум с небольшим коэффициентом


# Функция для случайной обрезки звука (random cropping)
def random_crop(data, crop_size=16000):
    if len(data) <= crop_size:
        return data  # Если аудио короче, возвращаем его как есть
    start = random.randint(0, len(data) - crop_size)
    return data[start:start + crop_size]


# Функция для затухания громкости (fade in/out)
def apply_fade(data, fade_in_duration=0.1, fade_out_duration=0.1, sr=16000):
    fade_in_samples = int(fade_in_duration * sr)
    fade_out_samples = int(fade_out_duration * sr)

    # Применяем линейное увеличение и уменьшение амплитуды
    data[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
    data[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)

    return data


def time_reverse(data):
    return data[::-1]


def change_volume(data, factor=1.5):
    return data * factor


# Основная функция для Data Augmentation
def augment_audio(file_path, output_path):
    try:
        # Загрузка аудио
        data, sr = librosa.load(file_path, sr=16000)

        # Генерация нескольких вариантов
        augmented_versions = [
            random_crop(data),  # Случайная обрезка
            apply_fade(data.copy()),  # Затухание громкости
            add_random_noise(data.copy()),  # Шум из внешних файлов
            add_random_noise(apply_fade(data.copy())),  # Комбинация шума и затухания
            time_reverse(data.copy()),
            change_volume(data.copy(), factor=0.8),  # Уменьшение громкости
            change_volume(data.copy(), factor=1.5),  # Увеличение громкости
        ]

        # Сохранение вариантов
        for i, augmented_data in enumerate(augmented_versions):
            augmented_file_path = os.path.join(output_path,
                                               f"{os.path.basename(file_path).replace('.wav', '')}_adv_aug_{i}.wav")
            sf.write(augmented_file_path, augmented_data, sr)
            print(f"Saved augmented file: {augmented_file_path}")

    except Exception as e:
        print(f"Failed to augment {file_path}: {e}")


# Применение Data Augmentation ко всем файлам в папке
for file_name in os.listdir(input_folder):
    if file_name.endswith(".wav"):
        file_path = os.path.join(input_folder, file_name)
        augment_audio(file_path, output_folder)
