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
---

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

# ⭐ Support

If this project helped you learn something or you found it useful, consider giving the repository a ⭐ on GitHub.

---