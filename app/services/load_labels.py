import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Завантаження CSV
df = pd.read_csv("app/data/big_labels.csv")

# Перетворення рядка "label1,label2,..." → список ['label1', 'label2']
df['labels'] = df['labels'].apply(lambda x: x.split(','))

# Бінаризація міток
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['labels'])

# Збереження класів (порядок важливий для декодування)
classes = mlb.classes_

# Перевірка розмірів
print(f"🔢 Кількість спектограм: {len(df)}")
print(f"🧷 Кількість класів: {len(classes)}")
print(f"📋 Класи: {classes}")
