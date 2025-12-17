import joblib
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text


class DummyModel:
    def predict(self, X):
        return [1.0]


def test_predict_saves_to_db(tmp_path, monkeypatch):
    # 1) Dummy model zapisany do pliku
    model_file = tmp_path / "dummy.pkl"
    joblib.dump(DummyModel(), model_file)

    # 2) ENV dla API (model + sqlite w pliku tymczasowym)
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    # 3) Import app po ustawieniu ENV
    from src.api.main import app

    client = TestClient(app)

    # 4) COUNT przed
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS predictions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts TEXT, payload TEXT, prediction REAL, model_version TEXT)"
            )
        )
        before = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar_one()

    # 5) Wywołanie /predict
    r = client.post("/predict", json={"feature_num": 2.0, "feature_cat": "A"})
    assert r.status_code == 200

    # 6) COUNT po
    with engine.begin() as conn:
        after = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar_one()

    assert after == before + 1
