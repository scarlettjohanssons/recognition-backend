import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Функция для создания мел-спектрограммы и сохранения её как изображения
def create_mel_spectrogram(file_path, save_path=None, sr=16000, n_mels=128):
    audio, _ = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if save_path:
        plt.figure(figsize=(3.2, 3.2))  # Размер изображения 224x224
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    return mel_spec_db

# Генерация мел-спектрограмм для всех классов
def generate_spectrograms(source_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(source_folder, file_name)
            save_path = os.path.join(target_folder, file_name.replace('.wav', '.png'))
            create_mel_spectrogram(file_path, save_path)

print("Генерация мел-спектрограмм...")
generate_spectrograms("app/data/ferrari_augmented", "app/data/mel_spectrograms/ferrari")
generate_spectrograms("app/data/audi_augmented", "app/data/mel_spectrograms/audi")
generate_spectrograms("app/data/unknown_wav", "app/data/mel_spectrograms/unknown")

# Настройка генераторов данных с расширенным Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    "app/data/mel_spectrograms",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    "app/data/mel_spectrograms",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Создание улучшенной модели CNN
print("Создание улучшенной модели...")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
print("Начало обучения улучшенной модели...")
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

# Сохранение модели
model_path = "app/models/cnn_classifier.h5"
model.save(model_path)
print(f"Модель успешно обучена и сохранена: {model_path}")

# Визуализация результатов обучения
def plot_training_results(history):
    plt.figure(figsize=(12, 6))

    # Точность
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Точность модели')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    # Потери
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Потери модели')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_results(history)
