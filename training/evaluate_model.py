from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.h5"
VALIDATION_DIR = BASE_DIR / "images" / "validation"

IMG_SIZE = (48, 48)
BATCH_SIZE = 128


# Load model
model = load_model(MODEL_PATH)


# Load validation images
datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_set = datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Predictions
predictions = model.predict(validation_set, verbose=1)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_set.classes

class_names = list(validation_set.class_indices.keys())


# Accuracy
accuracy = accuracy_score(true_classes, predicted_classes)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)


# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Facial Expression Recognition - Confusion Matrix")
plt.tight_layout()

plt.savefig(BASE_DIR / "confusion_matrix.png")

print("\nConfusion matrix saved as confusion_matrix.png")