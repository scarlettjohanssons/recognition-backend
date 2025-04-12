import os
import subprocess


def convert_mp3_to_wav(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith('.mp3'):
            source_file = os.path.join(source_folder, file_name)
            target_file = os.path.join(target_folder, file_name.replace('.mp3', '.wav'))

            if os.path.exists(target_file):
                print(f"Skipped (already converted): {target_file}")
                continue

            try:
                subprocess.run(["ffmpeg", "-i", source_file, target_file], check=True)
                print(f"Converted: {file_name} -> {target_file}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {file_name}: {e}")
