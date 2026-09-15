from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
import imageio_ffmpeg
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "onnx" / "resmasking_int8.onnx"
CASCADE_PATH = BASE_DIR / "HaarcascadeclassifierCascadeClassifier.xml"
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"

CAPTURED_DIR = BASE_DIR / "captured_images"
PREDICTED_DIR = BASE_DIR / "predicted_images"

CAPTURED_DIR.mkdir(exist_ok=True)
PREDICTED_DIR.mkdir(exist_ok=True)

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

app = FastAPI(title="Facial Expression Recognition - RMN")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"RMN model not found: {MODEL_PATH}\nRun: python download_model.py"
    )

if not CASCADE_PATH.exists():
    raise FileNotFoundError(f"Haar Cascade not found: {CASCADE_PATH}")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
face_classifier = cv2.CascadeClassifier(str(CASCADE_PATH))


def predict_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    results = []

    for x, y, w, h in faces:
        face = gray[y:y + h, x:x + w]

        if face.size == 0:
            continue

        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)

        tensor = np.stack([face, face, face], axis=0)
        tensor = tensor[np.newaxis].astype(np.float32) / 255.0

        logits = session.run(None, {input_name: tensor})[0][0]
        logits = logits - np.max(logits)

        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)

        emotion_index = int(np.argmax(probabilities))

        results.append({
            "expression": EMOTIONS[emotion_index],
            "confidence": round(float(probabilities[emotion_index]), 4),
            "box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            },
        })

    return results


def annotate_image(image, faces):
    output = image.copy()

    for index, face in enumerate(faces, start=1):
        box = face["box"]
        x, y = box["x"], box["y"]
        w, h = box["width"], box["height"]

        label = (
            f"Face {index}: {face['expression']} "
            f"{face['confidence'] * 100:.1f}%"
        )

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            output,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return output


@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "provider": session.get_providers()[0],
    }


@app.get("/video/{filename}")
def get_video(filename: str):
    path = PREDICTED_DIR / filename

    if not path.exists():
        raise HTTPException(404, "Video not found.")

    return FileResponse(path, media_type="video/mp4")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    source: str = "upload",
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not read the image.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{source}_{timestamp}.jpg"

    cv2.imwrite(str(CAPTURED_DIR / filename), image)

    faces = predict_faces(image)
    predicted_image = annotate_image(image, faces)
    cv2.imwrite(str(PREDICTED_DIR / filename), predicted_image)

    return {
        "filename": filename,
        "faces_detected": len(faces),
        "faces": faces,
    }


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Please select a video.")

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            400,
            "Supported videos: MP4, AVI, MOV, MKV and WEBM.",
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    input_name = f"upload_{timestamp}{suffix}"
    output_name = f"upload_{timestamp}_predicted.mp4"

    input_path = CAPTURED_DIR / input_name
    output_path = PREDICTED_DIR / output_name

    input_path.write_bytes(await file.read())

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise HTTPException(400, "Could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise HTTPException(400, "Could not read video dimensions.")

    # Browser-friendly H.264 MP4 output.
    # OpenCV's mp4v output may be created successfully but often
    # cannot be played directly by Chrome/Edge.
    output_width = width - (width % 2)
    output_height = height - (height % 2)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{output_width}x{output_height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    frame_number = 0
    last_faces = []

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if frame_number % 3 == 0:
                last_faces = predict_faces(frame)

            annotated = annotate_image(frame, last_faces)
            annotated = annotated[:output_height, :output_width]

            process.stdin.write(
                np.ascontiguousarray(annotated).tobytes()
            )

            frame_number += 1

    except (BrokenPipeError, OSError) as error:
        cap.release()
        process.kill()
        raise HTTPException(
            500,
            f"Video encoding failed: {error}",
        )

    finally:
        cap.release()

    process.stdin.close()
    stderr = process.stderr.read()
    return_code = process.wait()

    if return_code != 0:
        raise HTTPException(
            500,
            "Video encoding failed: " + stderr.decode(errors="ignore")[-1000:],
        )

    if frame_number == 0:
        raise HTTPException(400, "The video contains no readable frames.")

    return {
        "original_filename": input_name,
        "predicted_filename": output_name,
        "frames_processed": frame_number,
        "video_url": f"/video/{output_name}",
    }
