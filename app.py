# app.py
"""
FastAPI app for serving CNN trading signals.

Endpoints:
 - GET /health
 - POST /predict   -> returns a single prediction for the provided recent candle window
"""

import os
import traceback
from typing import Dict

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictRequest, PredictResponse
from utils_api import (
    df_from_pricepoints,
    prepare_features_for_inference,
    load_scaler,
    load_label_mapping,
    map_prediction_index_to_label,
)

app = FastAPI(title="CNN Trading Signal API")

# Allow browser-based testing if desired
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects loaded at startup
MODEL = None
SCALER = None
LABEL_MAPPING = None
MODEL_NAME = None
DEFAULT_TIMESTEPS = 30  # should match training pipeline


# 1) Load model
# Common saved locations: "models/best_cnn_model.keras" or "models/cnn_model"
model_paths = [
    "models/best_cnn_model.keras",
    "models/cnn_model",
    "models/best_cnn_model.h5",
    "models/cnn_model.keras",
]
loaded = False
for p in model_paths:
    if os.path.exists(p):
        try:
            MODEL = tf.keras.models.load_model(p)
            MODEL_NAME = p
            loaded = True
            break
        except Exception:
            # try next
            continue

if not loaded:
    # If no model found, leave MODEL = None. Endpoint will return 503.
    MODEL = None
    MODEL_NAME = None

# 2) Load scaler (joblib)
SCALER = load_scaler("models/scaler.joblib")

# 3) Load label mapping if available
LABEL_MAPPING = load_label_mapping("models/label_mapping.joblib")


@app.get("/health")
def health():
    ok = MODEL is not None
    return {"status": "ok" if ok else "model_not_loaded", "model_path": MODEL_NAME}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Accepts recent candle list (most recent last), runs preprocessing (same as training),
    runs the CNN model, and returns a trading signal mapped to {-1,0,1}, confidence, and class probabilities.
    """
    global MODEL, SCALER, LABEL_MAPPING

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    # Determine timesteps to use
    timesteps = req.timesteps if req.timesteps is not None else DEFAULT_TIMESTEPS

    # Convert incoming data to DataFrame
    raw_df = df_from_pricepoints([p.dict() for p in req.data])

    try:
        # Prepare features and shape them for inference
        X, feature_cols, feature_df = prepare_features_for_inference(raw_df, scaler=SCALER, timesteps=timesteps)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    # Run model prediction
    try:
        preds = MODEL.predict(X)  # shape (1, num_classes) or (1, 1)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    # Handle model output formats
    probs = None
    pred_index = None
    confidence = None

    preds = np.array(preds)

    if preds.ndim == 2 and preds.shape[1] > 1:
        probs = preds[0]  # vector of class probabilities
        pred_index = int(np.argmax(probs))
        confidence = float(np.max(probs))
    elif preds.ndim == 2 and preds.shape[1] == 1:
        # Regression / single-output -> convert to tanh in [-1,1] and threshold
        val = float(preds[0, 0])
        # Map to signals deterministically
        if val > 0.2:
            pred_label = 1
        elif val < -0.2:
            pred_label = -1
        else:
            pred_label = 0
        # Build fake probs
        probs = np.array([0.0, 0.0, 0.0])
        probs[ { -1: 0, 0: 1, 1: 2 }[pred_label] ] = 1.0
        pred_index = { -1: 0, 0: 1, 1: 2 }[pred_label]
        confidence = 1.0
    else:
        # Unexpected shape
        raise HTTPException(status_code=500, detail=f"Unexpected model output shape: {preds.shape}")

    # Map index to trading label (attempt to use mapping artifact if present)
    mapped_label = map_prediction_index_to_label(pred_index, LABEL_MAPPING)

    # Build class_probs dictionary in terms of trading labels
    # Default order assumed for 3-class: index 0->short, 1->hold, 2->long
    class_names = []
    if probs.shape[0] == 3:
        class_names = ["short", "hold", "long"]
    elif probs.shape[0] == 2:
        class_names = ["class_0", "class_1"]
    else:
        class_names = [f"class_{i}" for i in range(probs.shape[0])]

    class_probs = {name: float(probs[i]) for i, name in enumerate(class_names)}

    return PredictResponse(
        prediction=int(mapped_label),
        confidence=float(confidence),
        class_probs=class_probs,
        model_name=str(MODEL_NAME),
        timesteps_used=timesteps
    )