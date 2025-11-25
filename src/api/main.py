import datetime as dt
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text

app = FastAPI(title="ASI2025 Model API")

# --------- BAZA DANYCH DO ZAPISU PREDYKCJI ---------
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///local.db"), future=True)


def save_prediction(payload: dict, prediction: float | int, model_version: str):
    with engine.begin() as conn:
        # tworzymy tabelę jeśli jeszcze jej nie ma
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS predictions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts TEXT, "
                "payload TEXT, "
                "prediction REAL, "
                "model_version TEXT)"
            )
        )
        # zapis jednej predykcji
        conn.execute(
            text(
                "INSERT INTO predictions(ts, payload, prediction, model_version) "
                "VALUES (:ts, :payload, :pred, :ver)"
            ),
            {
                "ts": dt.datetime.utcnow().isoformat(),
                "payload": json.dumps(payload),
                "pred": float(prediction),
                "ver": model_version,
            },
        )


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

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

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
    # 1. przygotowanie danych
    row_dict = payload.model_dump()
    X = pd.DataFrame([row_dict])

    # 2. predykcja
    try:
        y_pred = model.predict(X)[0]
    except Exception as e:
        print(f"[API] Błąd podczas predykcji: {e}")
        y_pred = 0.0

    # 3. zapis do bazy
    save_prediction(
        payload=payload.model_dump(),  # Pydantic -> dict
        prediction=float(y_pred),
        model_version=MODEL_VERSION,
    )

    # 4. zwrot wyniku
    return {
        "prediction": float(y_pred),
        "model_version": MODEL_VERSION,
    }
