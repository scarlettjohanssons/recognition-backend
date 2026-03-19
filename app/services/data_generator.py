import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Конфігурація
csv_path = 'app/data/extended_labels.csv'
image_size = (224, 224)
batch_size = 16
validation_split = 0.2
seed = 42

# 1. Завантаження і перетворення
df = pd.read_csv(csv_path)
df['labels'] = df['labels'].apply(lambda x: x.split(','))

mlb = MultiLabelBinarizer()
multi_hot = mlb.fit_transform(df['labels'])
multi_hot_df = pd.DataFrame(multi_hot.astype('float32'), columns=mlb.classes_)
df = pd.concat([df[['filepath']], multi_hot_df], axis=1)
classes = mlb.classes_

# 2. Розділення
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=validation_split, random_state=seed, shuffle=True)

# 3. ImageDataGenerator з агресивною аугментацією
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 4. Генератори
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col=mlb.classes_.tolist(),
    target_size=image_size,
    class_mode='raw',
    batch_size=batch_size,
    shuffle=True,
    seed=seed
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col=mlb.classes_.tolist(),
    target_size=image_size,
    class_mode='raw',
    batch_size=batch_size,
    shuffle=False,
    seed=seed
)

# 5. Збереження класів
import pickle
with open('app/data/label_classes.pkl', 'wb') as f:
    pickle.dump(mlb.classes_.tolist(), f)

print("✅ Генератори готові. Класи збережені в app/data/label_classes.pkl")
