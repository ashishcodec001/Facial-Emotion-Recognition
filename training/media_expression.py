from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


BASE_DIR = Path(__file__).resolve().parent

# Just change this path to the photo or video you want to process.
SOURCE = BASE_DIR / "SAMPLE2.mp4"

MODEL_PATH = BASE_DIR / "model.h5"
FACE_CASCADE = BASE_DIR / "HaarcascadeclassifierCascadeClassifier.xml"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


face_classifier = cv2.CascadeClassifier(str(FACE_CASCADE))
classifier = load_model(MODEL_PATH)


def predict_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
    )

    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]

        if roi_gray.size == 0:
            continue

        roi_gray = cv2.resize(
            roi_gray,
            (48, 48),
            interpolation=cv2.INTER_AREA,
        )

        roi = roi_gray.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi, verbose=0)[0]

        emotion_index = int(np.argmax(prediction))
        emotion = EMOTIONS[emotion_index]
        confidence = float(prediction[emotion_index])

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2,
        )

        label = f"{emotion} ({confidence:.0%})"

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    return frame


def process_image():
    frame = cv2.imread(str(SOURCE))

    if frame is None:
        raise FileNotFoundError(f"Could not open image: {SOURCE}")

    frame = predict_faces(frame)

    cv2.imshow("Emotion Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video():
    cap = cv2.VideoCapture(str(SOURCE))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {SOURCE}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = predict_faces(frame)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"File not found: {SOURCE}")

    extension = SOURCE.suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        process_image()

    elif extension in VIDEO_EXTENSIONS:
        process_video()

    else:
        raise ValueError(
            f"Unsupported file type: {extension}\n"
            "Use a supported image or video format."
        )


if __name__ == "__main__":
    main()
