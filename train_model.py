
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = Path("dataset")

# Load training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["error", "normal"],
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["error", "normal"],
    shuffle=False
)

print("Class mapping:", train_ds.class_names)
# error = 0, normal = 1

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# CNN architecture
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Normalize RGB pixels from 0-255 to 0-1
    layers.Rescaling(1.0 / 255),

    # CNN feature extraction
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the model
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# Evaluate and save
loss, accuracy = model.evaluate(val_ds)
print(f"Validation accuracy: {accuracy:.2%}")

model.save("error_cnn.keras")
print("Model saved successfully.")
