# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time ASL (American Sign Language) letter classifier. A camera feed is captured in the browser, frames are sent to a FastAPI backend, and a TensorFlow CNN model returns the predicted letter (A–Z, excluding J and Z which require motion).

## Repository Layout

```
sign-language-classifier-app/   ← this repo
  backend/
    app.py                      ← FastAPI server (single file)
    requirements.txt
    Dockerfile                  ← targets Hugging Face Spaces (port 7860)
  frontend/
    index.html                  ← entire UI in one HTML file
```

A sibling directory `../sign-language-classifier-research/` contains the ML training code and trained `.keras` model files. The backend expects `best_model.keras` to be copied into `backend/` before running.

## Running Locally

**Backend:**
```bash
# Copy a trained model first
cp ../sign-language-classifier-research/models/mobilenet_best.keras backend/best_model.keras

cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 7860
```

Health check: `GET http://localhost:7860/` → `{"status": "ok", "model_loaded": true}`

**Frontend:**
```bash
cd frontend
python3 -m http.server 8000
# open http://localhost:8000
```

Before running the frontend, update the hardcoded backend URL in `frontend/index.html`:
```javascript
const BACKEND_URL = "https://your-space.hf.space";  // ← change this
```

**Docker:**
```bash
cd backend
docker build -t asl-classifier .
docker run -p 7860:7860 asl-classifier
```

## Architecture

**Prediction flow:**
1. Browser captures a video frame every 500 ms via `getUserMedia`
2. Frame is encoded as JPEG (quality 0.85) and base64'd
3. `POST /predict` sends `{"image": "<base64>"}` to the backend
4. Backend: base64 decode → grayscale → resize to 28×28 (LANCZOS) → normalize [0,1] → reshape to `(1,28,28,1)` → TF inference
5. Response: `{"letter": "A", "confidence": 0.9873}`

**Model:**
- Transfer learning on Sign Language MNIST (24 classes, 28×28 grayscale)
- Two trained options: MobileNetV2 (~11 MB) or ResNet50 (~92 MB)
- Model is loaded once at startup via FastAPI's lifespan context manager

**Label mapping** (`backend/app.py`): `LABELS = [chr(ord("A") + i) for i in range(26)]` — indices map directly to A–Z.

## Deployment (Hugging Face Spaces)

The Dockerfile is configured for HF Spaces:
- Port **7860**
- Non-root user `user` (UID 1000) — HF Spaces requirement
- Large model files tracked via Git LFS (`*.keras`)
