import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Конфігурація
data_dir = 'app/data/spectrograms'
img_height, img_width = 224, 224
batch_size = 32

# Генератори (знову створюємо їх!)
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1. / 255
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=42
)

# Завантаж раніше збережену модель
model = load_model('app/models/cnn_classifier.keras')

# Витягуємо базову MobileNetV2
base_model = model.layers[1]
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Компіляція з низьким learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Колбеки
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ModelCheckpoint('app/models/fine_tuned_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
]

# Тренування
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks
)

# Збереження
model.save('app/models/final_finetuned_model.keras', save_format="keras")
print("✅ Fine-tuned модель збережено!")
