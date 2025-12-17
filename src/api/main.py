from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.api.settings import settings

app = FastAPI(title="ASI2025 Model API")


# --------- SCHEMAT WEJŚCIA ---------
class Features(BaseModel):
    feature_num: float
    feature_cat: str


# --------- SCHEMAT WYJŚCIA ---------
class Prediction(BaseModel):
    prediction: float
    model_version: str


# --------- ŁADOWANIE MODELU (OPCJA A – PLIK LOKALNY) ---------
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BASE_DIR / "data" / "06_models" / "ag_production.pkl"

# Zamiast os.getenv(...) bierzemy z Settings (env/.env)
MODEL_PATH = Path(settings.MODEL_PATH or str(DEFAULT_MODEL_PATH))
MODEL_VERSION = f"file:{MODEL_PATH.name}"

print(f"[API] Ładuję model z: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print("[API] Model załadowany OK")
except Exception as e:
    raise RuntimeError(f"Nie udało się załadować modelu z {MODEL_PATH}: {e}")


# --------- ENDPOINT HEALTHCHECK ---------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# --------- ENDPOINT PREDICT ---------
@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    """
    1. Zamiana Pydantic -> dict -> DataFrame
    2. Wywołanie model.predict(...)
    3. Zwrócenie predykcji + wersji modelu
    """
    row_dict = payload.model_dump()
    X = pd.DataFrame([row_dict])

    try:
        y_pred = model.predict(X)[0]
    except Exception as e:
        print(f"[API] Błąd podczas predykcji: {e}")
        y_pred = 0.0

    return {
        "prediction": float(y_pred),
        "model_version": MODEL_VERSION,
    }
