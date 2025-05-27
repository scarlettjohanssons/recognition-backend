import os
import shutil
import librosa

def copy_fixed_duration_audio():
    # === Змінні ===
    source_dir = '...'
    target_dir = '...'
    duration_sec = 5.0                       # Тривалість у секундах
    tolerance = 0.1                          # Допустиме відхилення

    # === Створення цільової директорії ===
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    matched = 0
    total = 0

    for root, _, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        target_subdir = os.path.join(target_dir, rel_path)
        os.makedirs(target_subdir, exist_ok=True)

        for file in files:
            if not file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                continue

            file_path = os.path.join(root, file)
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = len(y) / sr
            except Exception as e:
                print(f"⚠️ Проблема з файлом {file_path}: {e}")
                continue

            total += 1
            if abs(duration - duration_sec) <= tolerance:
                shutil.copy2(file_path, os.path.join(target_subdir, file))
                matched += 1

    print(f"✅ Знайдено {matched} з {total} файлів ≈ {duration_sec:.2f} сек")

# Викликати функцію
copy_fixed_duration_audio()
