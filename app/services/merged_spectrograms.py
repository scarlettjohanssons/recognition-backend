import os
import shutil
import pandas as pd
from collections import defaultdict

# Абсолютні або відносні шляхи
source_dir = 'app/data/big_mel_spectrograms'
target_dir = 'app/data/big_merged_spectrograms'
os.makedirs(target_dir, exist_ok=True)

file_labels = defaultdict(set)

# 1. Сканування джерела
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for fname in os.listdir(class_path):
        if not fname.lower().endswith('.png'):
            continue

        src_path = os.path.join(class_path, fname)
        dest_path = os.path.join(target_dir, fname)

        # Якщо файл уже існує — перейменовуємо
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(fname)
            i = 1
            while os.path.exists(os.path.join(target_dir, f"{base}_{i}{ext}")):
                i += 1
            dest_path = os.path.join(target_dir, f"{base}_{i}{ext}")
            fname = f"{base}_{i}{ext}"

        shutil.copy2(src_path, dest_path)
        file_labels[fname].add(class_name)

# 2. Створення CSV
data = [{'filepath': os.path.join(target_dir, fname), 'labels': ','.join(sorted(labels))}
        for fname, labels in file_labels.items()]

df = pd.DataFrame(data)
df.to_csv('app/data/big_labels.csv', index=False)

print(f"✅ Завершено: {len(df)} файлів оброблено. CSV збережено в app/data/labels.csv")
