from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ASI2025 Model API")


# --------- SCHEMAT WEJŚCIA ---------
class Features(BaseModel):
    feature_num: float
    feature_cat: str


# --------- SCHEMAT WYJŚCIA ---------
class Prediction(BaseModel):
    prediction: float
    model_version: str


# --------- ENDPOINT HEALTHCHECK ---------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# --------- ENDPOINT PREDICT (NA RAZIE NA SZTYWNO) ---------
@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    return {
        "prediction": 0.0,
        "model_version": "local-dev",
    }
