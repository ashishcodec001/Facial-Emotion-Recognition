# Facial Expression Recognition - RMN

RMN-based facial expression recognition app with FastAPI, OpenCV, ONNX Runtime and a browser UI.

## Features

- One upload section for images and videos
- Webcam start/stop toggle
- Camera-icon capture button
- Multiple face detection
- Seven emotions
- Confidence score
- Original images/videos saved in `captured_images/`
- Predicted images/videos saved in `predicted_images/`
- CPU inference with RMN INT8 ONNX model
- Browser-friendly H.264 MP4 video output

## Why FFmpeg is used

The previous version used OpenCV's `mp4v` codec. That can produce a valid MP4 file that Chrome/Edge cannot play.

This version uses `imageio-ffmpeg` to encode the predicted video as H.264 (`libx264`) with `yuv420p`, which is much more compatible with browsers.

## Run

```powershell
pip install -r requirements.txt
python -m uvicorn app:app
```

Open:

```text
http://127.0.0.1:8000
```

Use an MP4 video first for testing.
