import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from app.services.data_generator import train_generator, val_generator

# 1. Завантаження моделі
model = load_model('app/models/multi_best_model.keras')

# 2. Повторна компіляція з меншою швидкістю навчання
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # менш агресивне донавчання
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 3. Колбеки
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ModelCheckpoint('app/models/multi_finetuned_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
]

# 4. Навчання
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks
)

# 5. Збереження
model.save('app/models/multi_finetuned_model.keras')
print("✅ Фінетюнінг завершено і модель збережена.")
