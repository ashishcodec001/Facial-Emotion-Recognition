from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


BASE_DIR = Path(__file__).resolve().parent

FACE_CASCADE = BASE_DIR / "HaarcascadeclassifierCascadeClassifier.xml"
MODEL_PATH = BASE_DIR / "model.h5"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]


face_classifier = cv2.CascadeClassifier(str(FACE_CASCADE))
classifier = load_model(MODEL_PATH)


def main():
    if face_classifier.empty():
        raise FileNotFoundError(f"Could not load face cascade: {FACE_CASCADE}")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read a frame from the webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) == 0:
                cv2.putText(
                    frame,
                    "No Faces",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
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

            cv2.imshow("Emotion Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
