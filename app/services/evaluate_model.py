import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from app.services.data_generator import val_generator  # використовуй той же генератор

# 1. Завантаження моделі та класів
model = load_model('app/models/multi_finetuned_model.keras')
with open('app/data/label_classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# 2. Прогноз
y_true = val_generator.labels
y_pred = model.predict(val_generator, verbose=1)
y_pred_bin = (y_pred > 0.5).astype(int)

# 3. Класифікаційний звіт
report = classification_report(y_true, y_pred_bin, target_names=classes, output_dict=True)
df_report = pd.DataFrame(report).transpose()
print(df_report[['precision', 'recall', 'f1-score']])

# 4. Heatmap F1-score по класах
plt.figure(figsize=(10, 6))
sns.heatmap(df_report.loc[classes][['f1-score']], annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
plt.title("F1-score per class")
plt.ylabel("Class")
plt.xlabel("F1-score")
plt.tight_layout()
plt.show()
