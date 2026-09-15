from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "images" / "train"
VALIDATION_DIR = BASE_DIR / "images" / "validation"

MODEL_PATH = BASE_DIR / "model.h5"


# Image settings
IMG_SIZE = (48, 48)
BATCH_SIZE = 128
NUM_CLASSES = 7


# Check dataset
if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}"
    )

if not VALIDATION_DIR.exists():
    raise FileNotFoundError(
        f"Validation folder not found: {VALIDATION_DIR}"
    )


# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
)


# Validation data should not be augmented
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


train_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
)


validation_set = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
)


print("\nClass indices:")
print(train_set.class_indices)


# Calculate class weights
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_set.classes),
    y=train_set.classes,
)

class_weights = {
    i: weight
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights:")
print(class_weights)


# CNN model
model = Sequential([
    Conv2D(
        64,
        (3, 3),
        padding="same",
        input_shape=(48, 48, 1),
    ),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(
        128,
        (5, 5),
        padding="same",
    ),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(
        512,
        (3, 3),
        padding="same",
    ),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(
        512,
        (3, 3),
        padding="same",
    ),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(256),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.25),

    Dense(512),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.25),

    Dense(NUM_CLASSES, activation="softmax"),
])


# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


model.summary()


# Save the best model automatically
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1,
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1,
)


# Train
history = model.fit(
    train_set,
    validation_data=validation_set,
    epochs=40,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
    ],
)


# Load the best weights selected by ModelCheckpoint
model.load_weights(MODEL_PATH)


# Final validation score
loss, accuracy = model.evaluate(
    validation_set,
    verbose=1,
)

print("\nBest model validation accuracy:")
print(f"{accuracy * 100:.2f}%")


# Save training graph
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    BASE_DIR / "training_accuracy.png"
)

plt.close()


print("\nTraining completed.")
print(f"Best model saved to: {MODEL_PATH}")
print("Training graph saved as: training_accuracy.png")
