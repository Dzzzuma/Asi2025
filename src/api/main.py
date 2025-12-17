from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.api.db import save_prediction
from src.api.settings import get_settings

app = FastAPI(title="ASI2025 Model API")


# --------- SCHEMAT WEJŚCIA ---------
class Features(BaseModel):
    feature_num: float
    feature_cat: str


# --------- SCHEMAT WYJŚCIA ---------
class Prediction(BaseModel):
    prediction: float
    model_version: str


# --------- ŚCIEŻKA DOMYŚLNA MODELU ---------
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BASE_DIR / "data" / "06_models" / "ag_production.pkl"

# Model jako singleton – ładowany dopiero przy pierwszej predykcji
_MODEL = None
_MODEL_VERSION = None


def get_model_and_version():
    global _MODEL, _MODEL_VERSION

    if _MODEL is None:
        settings = get_settings()
        model_path = Path(settings.MODEL_PATH or str(DEFAULT_MODEL_PATH))
        _MODEL_VERSION = f"file:{model_path.name}"

        print(f"[API] Ładuję model z: {model_path}")
        _MODEL = joblib.load(model_path)
        print("[API] Model załadowany OK")

    return _MODEL, _MODEL_VERSION


# --------- ENDPOINT HEALTHCHECK ---------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# --------- ENDPOINT PREDICT ---------
@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    row_dict = payload.model_dump()
    X = pd.DataFrame([row_dict])

    model, model_version = get_model_and_version()

    try:
        y_pred = model.predict(X)[0]
    except Exception as e:
        print(f"[API] Błąd podczas predykcji: {e}")
        y_pred = 0.0

    save_prediction(payload=row_dict, prediction=float(y_pred), model_version=model_version)

    return {"prediction": float(y_pred), "model_version": model_version}
