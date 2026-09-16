# 🧠 Facial Expression Recognition

<p align="center">
  <strong>AI-powered facial expression recognition using a Residual Masking Network (RMN), ONNX Runtime, OpenCV, FastAPI, and a modern web interface.</strong>
</p>

<p align="center">
  <a href="https://github.com/ashishcode001/Facial-Emotion-Recognition">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <a href="https://facial-emotion-recognition-glm5.onrender.com">
    <img src="https://img.shields.io/badge/Live-Demo-46E3B7?style=for-the-badge&logo=render" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
</p>

---

## ✨ Overview

**Facial Expression Recognition** is a computer-vision application that detects human faces in images and videos and predicts the facial expression of each detected face.

The application combines:

- 👤 **Haar Cascade** face detection
- 🧠 **Residual Masking Network (RMN)** for expression classification
- ⚡ **ONNX Runtime** for CPU-based inference
- 🖼️ **OpenCV** for image/video processing
- 🚀 **FastAPI** for the backend API
- 🌐 **HTML/CSS/JavaScript** for the web interface
- 🐳 **Docker** for containerized deployment
- ☁️ **Render** for cloud deployment
- ☁️ **Azure Blob Storage** for cloud-based file storage

The system supports **image uploads, video uploads, and webcam capture**.

> **Live Demo:** https://facial-emotion-recognition-glm5.onrender.com

> **Source Code:** https://github.com/ashishcode001/Facial-Emotion-Recognition

---

## 🎯 What This Project Does

The application follows a simple computer-vision pipeline:

```text
Image / Video / Webcam
          │
          ▼
   Face Detection
   (Haar Cascade)
          │
          ▼
   Face Extraction
          │
          ▼
     Preprocessing
     224 × 224 RGB
          │
          ▼
       RMN Model
     ONNX Runtime
          │
          ▼
 Emotion Classification
          │
          ▼
 Expression + Confidence
          │
          ▼
 Annotated Result
```

For every detected face, the application returns:

- Expression label
- Confidence score
- Face bounding box
- Annotated image/video

---

# 🚀 Features

### 🖼️ Image Recognition

Upload an image and the application will:

1. Read the image.
2. Detect faces.
3. Extract each face.
4. Preprocess the face.
5. Run RMN inference.
6. Predict the expression.
7. Draw a bounding box and prediction label.
8. Return the prediction results.

### 🎥 Video Recognition

Upload supported video formats:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`

The application processes the video frame-by-frame, performs facial-expression inference at selected intervals, carries forward the latest prediction between inference frames, and generates a browser-friendly MP4 result.

### 📷 Webcam Recognition

The web interface supports browser webcam access.

Workflow:

```text
Start Webcam
     ↓
Camera Preview
     ↓
Capture
     ↓
Send Frame to API
     ↓
Face Detection + RMN
     ↓
Prediction
     ↓
Display Result
```

The webcam capture is processed by the same backend prediction pipeline used for uploaded images.

### 📊 Confidence Scores

Each detected face receives a predicted expression and confidence score.

Example:

```text
Face 1: Happy — 94.2%
Face 2: Neutral — 81.7%
```

### 👥 Multiple Face Detection

The application can detect multiple faces in a single image/frame and produce an individual prediction for each detected face.

### ⚡ CPU Inference

The RMN model runs through:

```text
ONNX Runtime
      ↓
CPUExecutionProvider
```

This allows the project to run without requiring a dedicated NVIDIA GPU.

### 🌐 Web Interface

The frontend provides:

- Image upload
- Video upload
- Webcam access
- Capture button
- Prediction display
- Video result playback
- Expression and confidence information
- Responsive interface

### 🐳 Docker Support

The project includes a Docker configuration so the application can be packaged with its runtime dependencies and started consistently across environments.

### ☁️ Cloud Deployment

The application is configured for deployment on Render.

The deployed application is publicly accessible through an HTTPS URL.

### ☁️ Cloud Storage

The project is designed to use Azure Blob Storage for storing captured/original and predicted media instead of depending exclusively on the local filesystem.

---

# 🧠 Supported Expressions

The model uses **7 expression classes**:

| ID | Expression |
|---:|---|
| 0 | 😠 Angry |
| 1 | 🤢 Disgust |
| 2 | 😨 Fear |
| 3 | 😊 Happy |
| 4 | 😢 Sad |
| 5 | 😲 Surprise |
| 6 | 😐 Neutral |

---

# 🏗️ Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Computer Vision | OpenCV |
| Face Detection | Haar Cascade |
| Deep Learning Model | Residual Masking Network (RMN) |
| Model Format | ONNX |
| Inference | ONNX Runtime |
| Numerical Processing | NumPy |
| Video Encoding | FFmpeg via `imageio-ffmpeg` |
| Frontend | HTML, CSS, JavaScript |
| Containerization | Docker |
| Deployment | Render |
| Cloud Storage | Azure Blob Storage |
| Model Hosting | Hugging Face |

---

# 🧠 Model

This project uses a **Residual Masking Network (RMN)** model in ONNX format.

Model file:

```text
models/onnx/resmasking_int8.onnx
```

The model is downloaded automatically during the Docker image build using:

```text
download_model.py
```

This avoids committing the large model binary directly to the Git repository.

### Model Source

The RMN model is hosted on Hugging Face:

https://huggingface.co/phamquiluan/ResidualMaskingNetwork

### Input Processing

For each detected face:

```text
Original Image
      ↓
Convert BGR → Grayscale
      ↓
Crop Face
      ↓
Resize → 224 × 224
      ↓
Create 3-channel tensor
      ↓
Normalize to [0, 1]
      ↓
RMN / ONNX Runtime
```

The prediction output is converted into probabilities and the class with the highest probability is selected.

---

# 📁 Project Structure

```text
Facial-Emotion-Recognition/
│
├── captured_images/
│   └── Original/captured media
│
├── predicted_images/
│   └── Annotated prediction media
│
├── frontend/
│   └── index.html
│
├── models/
│   └── onnx/
│       └── resmasking_int8.onnx
│
├── training/
│   └── Training-related files
│
├── app.py
│   └── FastAPI application
│
├── download_model.py
│   └── Downloads the RMN model
│
├── Dockerfile
│   └── Docker image configuration
│
├── requirements.txt
│   └── Python dependencies
│
├── .dockerignore
│   └── Docker build exclusions
│
├── .gitignore
│   └── Git exclusions
│
├── HaarcascadeclassifierCascadeClassifier.xml
│   └── Haar Cascade face detector
│
└── README.md
```

---

# 🔌 API Endpoints

## `GET /`

Returns the web application frontend.

---

## `GET /health`

Health-check endpoint.

Example:

```json
{
  "status": "ok",
  "model": "resmasking_int8.onnx",
  "provider": "CPUExecutionProvider"
}
```

Useful for checking whether the backend and model are available.

---

## `POST /predict`

Predict facial expressions from an uploaded image.

### Request

```text
multipart/form-data
file=<image>
source=<upload/webcam>
```

### Example Response

```json
{
  "filename": "webcam_20260916_123456_123456.jpg",
  "faces_detected": 1,
  "faces": [
    {
      "expression": "Happy",
      "confidence": 0.9421,
      "box": {
        "x": 120,
        "y": 80,
        "width": 210,
        "height": 210
      }
    }
  ]
}
```

---

## `POST /predict-video`

Processes an uploaded video and returns the generated prediction video URL.

Supported formats:

```text
MP4
AVI
MOV
MKV
WEBM
```

Example response:

```json
{
  "original_filename": "upload_20260916_123456.mp4",
  "predicted_filename": "upload_20260916_123456_predicted.mp4",
  "frames_processed": 300,
  "video_url": "/video/upload_20260916_123456_predicted.mp4"
}
```

---

## `GET /video/{filename}`

Serves the generated prediction video as an MP4 file.

---

# 🧩 Backend Architecture

The backend is implemented using FastAPI.

The main processing functions are conceptually divided into:

```text
predict_faces()
      │
      ├── Face Detection
      ├── Face Cropping
      ├── Resize
      ├── Tensor Preparation
      ├── ONNX Inference
      └── Emotion Classification

annotate_image()
      │
      ├── Bounding Box
      ├── Face Number
      ├── Expression
      └── Confidence

/predict
      │
      └── Image/Webcam API

/predict-video
      │
      └── Video API
```

---

# 🛠️ Local Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ashishcode001/Facial-Emotion-Recognition.git
```

```bash
cd Facial-Emotion-Recognition
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download the Model

```bash
python download_model.py
```

The model should be available at:

```text
models/onnx/resmasking_int8.onnx
```

---

## 5. Start the Application

```bash
python -m uvicorn app:app --reload
```

Open:

```text
http://localhost:8000
```

---

# 🐳 Run with Docker

## Build the Image

```bash
docker build -t facial-expression-recognition .
```

## Run the Container

```bash
docker run --rm -e PORT=10000 -p 10000:10000 facial-expression-recognition
```

Open:

```text
http://localhost:10000
```

### Why Docker?

Docker packages:

- Python runtime
- Python dependencies
- OpenCV dependencies
- ONNX Runtime
- FFmpeg
- Application code
- Model download process

into a reproducible deployment environment.

---

# ☁️ Render Deployment

The project can be deployed as a Render Web Service.

Typical configuration:

```text
Environment: Docker
Port: 10000
Host: 0.0.0.0
```

The Docker container starts the application using:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
```

The deployed application is available here:

https://facial-emotion-recognition-glm5.onrender.com

### Render Notes

The free Render environment can spin down after inactivity. Therefore, the first request after a period of inactivity may take longer while the service starts again.

---

# ☁️ Azure Blob Storage

For cloud deployments, storing application media only on the local filesystem is not reliable because container filesystems can be ephemeral.

The project can therefore use **Azure Blob Storage** for persistent cloud storage.

Recommended structure:

```text
emotion-recognition/
│
├── captured/
│   ├── webcam_*.jpg
│   └── upload_*.jpg
│
└── predicted/
    ├── webcam_*.jpg
    └── upload_*_predicted.mp4
```

Environment variables:

```text
AZURE_STORAGE_CONNECTION_STRING=<your-secret-connection-string>
AZURE_CONTAINER_NAME=emotion-recognition
```

> **Security:** Never commit the Azure connection string to GitHub. Store it in environment variables/secrets.

---

# 🔐 Security Considerations

This application processes facial images, which can contain sensitive personal information.

Recommended practices:

- Do not commit cloud credentials.
- Do not commit `.env` files.
- Keep Azure Blob Storage containers private unless public access is intentionally required.
- Use short-lived SAS URLs when browser access to private blobs is required.
- Avoid storing unnecessary personal images.
- Delete old files when they are no longer needed.
- Restrict cloud-storage permissions to the minimum required.
- Do not expose storage account keys in frontend JavaScript.

---

# ⚡ Performance

Facial-expression recognition involves multiple CPU-heavy operations:

```text
Video decoding
      +
Face detection
      +
Face preprocessing
      +
ONNX inference
      +
Frame annotation
      +
Video encoding
```

Image inference is generally much faster than processing a complete video because video processing requires many frames to be decoded, analyzed, annotated, and encoded.

The video pipeline therefore uses inference at selected frame intervals while retaining the latest prediction between inference frames. This reduces unnecessary model calls while keeping the generated video at its original playback frame rate.

---

# 🎥 Video Processing Pipeline

```text
Uploaded Video
      │
      ▼
OpenCV VideoCapture
      │
      ▼
Read Frame
      │
      ├──── Inference Frame ────► Face Detection
      │                              │
      │                              ▼
      │                         RMN Prediction
      │                              │
      │                              ▼
      │                         Latest Result
      │
      └──── Other Frame ─────────► Reuse Latest Result
                                     │
                                     ▼
                              Annotate Frame
                                     │
                                     ▼
                              FFmpeg / H.264
                                     │
                                     ▼
                              Output MP4
```

---

# 🖥️ Using the Application

## Image

1. Open the application.
2. Select an image.
3. Click **Predict**.
4. Wait for processing.
5. View the detected faces and expression predictions.

## Video

1. Select a supported video.
2. Click **Predict**.
3. Wait for processing.
4. Play the generated prediction video.

## Webcam

1. Click **Start Webcam**.
2. Allow browser camera permission.
3. Position your face inside the camera view.
4. Click **Capture**.
5. Wait for the prediction.
6. View the annotated result.

---

# 📦 Dependencies

The project uses the following main Python packages:

```text
fastapi
uvicorn
python-multipart
numpy
opencv-contrib-python-headless
onnxruntime
huggingface_hub
imageio-ffmpeg
```

---

# 🧪 Example Use Cases

This project can be used as a foundation for:

- 🎓 Academic computer-vision projects
- 🧠 Deep-learning demonstrations
- 👨‍💻 AI/ML portfolio projects
- 📷 Webcam-based emotion experiments
- 🎥 Video expression analysis
- 🔬 Human-computer interaction research prototypes
- 🌐 AI inference API demonstrations
- ☁️ Cloud deployment practice
- 🐳 Docker deployment practice

---

# ⚠️ Limitations

Facial expression recognition is inherently challenging.

Predictions can be affected by:

- Lighting conditions
- Camera quality
- Face angle
- Occlusion
- Facial distance from camera
- Image resolution
- Facial expression intensity
- Multiple faces
- Background conditions
- Dataset/model limitations

The predicted label represents the model's classification and should **not** be interpreted as a reliable measurement of a person's actual emotional or mental state.

---

# 📈 Future Improvements

Possible improvements include:

- [ ] Real-time continuous webcam inference
- [ ] Face tracking between video frames
- [ ] Faster face detection
- [ ] GPU inference support
- [ ] Better video-processing optimization
- [ ] Progress bar for long videos
- [ ] Drag-and-drop uploads
- [ ] Prediction history
- [ ] Cloud media gallery
- [ ] Authentication
- [ ] Automatic cloud-file cleanup
- [ ] Model comparison
- [ ] More advanced face detectors
- [ ] Confidence threshold controls
- [ ] Mobile UI improvements
- [ ] Automated testing
- [ ] CI/CD pipeline

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

### Computer Vision

- Face detection
- Image preprocessing
- Bounding boxes
- Video frame processing
- Webcam capture

### Deep Learning

- Pre-trained model inference
- Multi-class classification
- Softmax probabilities
- Model input preprocessing
- ONNX model deployment

### Backend Development

- REST APIs
- FastAPI
- File uploads
- JSON responses
- API health checks

### Frontend Development

- Browser camera APIs
- JavaScript `fetch`
- FormData
- Image/video rendering
- Async processing states

### DevOps / Deployment

- Docker
- Environment variables
- Cloud deployment
- Render
- Azure Blob Storage

---

# 🗺️ Architecture at a Glance

```text
                         ┌─────────────────────┐
                         │      User           │
                         └──────────┬──────────┘
                                    │
                    Image / Video / Webcam
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Web Frontend      │
                         │ HTML/CSS/JavaScript  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │       Backend       │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
             ┌───────────────┐             ┌───────────────┐
             │  OpenCV Haar  │             │  ONNX Runtime │
             │ Face Detection│             │      RMN      │
             └───────┬───────┘             └───────┬───────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Expression +        │
                         │ Confidence + Box    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Annotated Result    │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  Web Interface          Azure Storage
```

---

# 📜 License

Add the appropriate license for your project before distributing it publicly.

If this repository is intended as a personal portfolio/academic project, clearly state the usage permissions you want to provide.

---

# 👨‍💻 Author

**Ashish**

AI/ML • Data Science • Computer Vision • Backend Development

GitHub:

https://github.com/ashishcode001

Project Repository:

https://github.com/ashishcode001/Facial-Emotion-Recognition

---

# ⭐ Support

If this project helped you learn something or you found it useful, consider giving the repository a ⭐ on GitHub.

---

<p align="center">
  <strong>Built with Python, OpenCV, RMN, ONNX Runtime, FastAPI, Docker, Render & Azure ☁️</strong>
</p>
