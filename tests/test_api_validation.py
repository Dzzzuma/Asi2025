from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_validation_error_returns_422():
    # feature_num powinien być float, a dajemy string -> FastAPI/Pydantic zwróci 422
    r = client.post("/predict", json={"feature_num": "oops", "feature_cat": "A"})
    assert r.status_code == 422
