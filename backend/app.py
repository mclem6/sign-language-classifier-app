import base64
import io
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import tensorflow as tf

LABELS = [chr(ord("A") + i) for i in range(26)]  # A–Z, index 0–25

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = tf.keras.models.load_model("mobilenet_best.keras")
    yield


app = FastAPI(title="ASL Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class Frame(BaseModel):
    # Pure base64 string (no data-URL prefix)
    image: str


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(frame: Frame):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Strip data-URL prefix if the client accidentally included it
    raw = frame.image
    if "," in raw:
        raw = raw.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw)
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
        img = img.resize((28, 28), Image.LANCZOS)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))

    return {"letter": LABELS[idx], "confidence": round(float(probs[idx]), 4)}
