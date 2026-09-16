from pathlib import Path
from datetime import datetime, timedelta
import os

import cv2
import numpy as np
import onnxruntime as ort
import imageio_ffmpeg
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
    BlobSasPermissions,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "onnx" / "resmasking_int8.onnx"
CASCADE_PATH = BASE_DIR / "HaarcascadeclassifierCascadeClassifier.xml"
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"

# These directories are used only as temporary working space for video processing.
# Uploaded images/videos are stored in Azure Blob Storage.
CAPTURED_DIR = BASE_DIR / "captured_images"
PREDICTED_DIR = BASE_DIR / "predicted_images"
CAPTURED_DIR.mkdir(exist_ok=True)
PREDICTED_DIR.mkdir(exist_ok=True)

AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "emotion-recognition")

if not AZURE_CONNECTION_STRING:
    raise RuntimeError(
        "AZURE_STORAGE_CONNECTION_STRING environment variable is not set. "
        "Add it to your local environment or Render environment variables."
    )

try:
    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_CONNECTION_STRING
    )
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
    # Verify the configured container is reachable.
    container_client.get_container_properties()
except Exception as exc:
    raise RuntimeError(
        f"Could not connect to Azure Blob Storage container '{AZURE_CONTAINER_NAME}': {exc}"
    ) from exc

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


def upload_bytes_to_azure(data: bytes, blob_name: str, content_type: str) -> str:
    """Upload bytes to Azure Blob Storage and return a temporary SAS URL."""
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )
    return create_blob_sas_url(blob_name)


def upload_file_to_azure(file_path: Path, blob_name: str, content_type: str) -> str:
    """Upload a local temporary file to Azure and return a temporary SAS URL."""
    blob_client = container_client.get_blob_client(blob_name)
    with file_path.open("rb") as data:
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
    return create_blob_sas_url(blob_name)


def _get_account_credentials():
    """Read account name/key from the configured Azure connection string."""
    values = {}
    for part in AZURE_CONNECTION_STRING.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value

    account_name = values.get("AccountName")
    account_key = values.get("AccountKey")
    if not account_name or not account_key:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING must contain AccountName and AccountKey."
        )
    return account_name, account_key


def create_blob_sas_url(blob_name: str, expiry_minutes: int = 60) -> str:
    """Create a short-lived read-only URL for a private Azure blob."""
    account_name, account_key = _get_account_credentials()
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=AZURE_CONTAINER_NAME,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=expiry_minutes),
    )
    return (
        f"https://{account_name}.blob.core.windows.net/"
        f"{AZURE_CONTAINER_NAME}/{blob_name}?{sas_token}"
    )


def redirect_to_blob(blob_name: str):
    """Redirect browser requests to a temporary SAS URL for a private blob."""
    try:
        blob_client = container_client.get_blob_client(blob_name)
        if not blob_client.exists():
            raise HTTPException(404, "File not found.")
        return RedirectResponse(create_blob_sas_url(blob_name))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Could not access Azure Blob Storage: {exc}") from exc


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
        "storage": "azure_blob",
        "container": AZURE_CONTAINER_NAME,
    }


# Backward-compatible media routes. The frontend can continue requesting these
# paths even though the actual files now live in Azure Blob Storage.
@app.get("/captured_images/{filename:path}")
def get_captured_image(filename: str):
    safe_name = Path(filename).name
    return redirect_to_blob(f"captured/{safe_name}")


@app.get("/predicted_images/{filename:path}")
def get_predicted_image(filename: str):
    safe_name = Path(filename).name
    return redirect_to_blob(f"predicted/{safe_name}")


@app.get("/video/{filename:path}")
def get_video(filename: str):
    safe_name = Path(filename).name
    return redirect_to_blob(f"predicted/{safe_name}")


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

    faces = predict_faces(image)
    predicted_image = annotate_image(image, faces)

    ok_original, original_encoded = cv2.imencode(".jpg", image)
    ok_predicted, predicted_encoded = cv2.imencode(".jpg", predicted_image)

    if not ok_original or not ok_predicted:
        raise HTTPException(500, "Could not encode the processed image.")

    try:
        captured_url = upload_bytes_to_azure(
            original_encoded.tobytes(),
            f"captured/{filename}",
            "image/jpeg",
        )
        predicted_url = upload_bytes_to_azure(
            predicted_encoded.tobytes(),
            f"predicted/{filename}",
            "image/jpeg",
        )
    except Exception as exc:
        raise HTTPException(500, f"Azure upload failed: {exc}") from exc

    return {
        "filename": filename,
        "faces_detected": len(faces),
        "faces": faces,
        "captured_url": captured_url,
        "predicted_url": predicted_url,
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

    try:
        input_path.write_bytes(await file.read())

        # Store the original uploaded video in Azure.
        captured_video_url = upload_file_to_azure(
            input_path,
            f"captured/{input_name}",
            file.content_type or "application/octet-stream",
        )

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
            ) from error
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

        predicted_video_url = upload_file_to_azure(
            output_path,
            f"predicted/{output_name}",
            "video/mp4",
        )

        return {
            "original_filename": input_name,
            "predicted_filename": output_name,
            "frames_processed": frame_number,
            "video_url": predicted_video_url,
            "captured_video_url": captured_video_url,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Video processing/upload failed: {exc}") from exc
    finally:
        # Local files are temporary only. Azure is the persistent storage layer.
        try:
            if input_path.exists():
                input_path.unlink()
        except OSError:
            pass

        try:
            if output_path.exists():
                output_path.unlink()
        except OSError:
            pass
