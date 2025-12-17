from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache

from sqlalchemy import create_engine, text

from src.api.settings import get_settings


@lru_cache
def get_engine():
    settings = get_settings()
    return create_engine(settings.DATABASE_URL, future=True)


def init_db() -> None:
    engine = get_engine()
    backend = engine.url.get_backend_name()

    if backend == "sqlite":
        ddl = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            payload TEXT,
            prediction REAL,
            model_version TEXT
        )
        """
    else:
        ddl = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP,
            payload JSONB,
            prediction DOUBLE PRECISION,
            model_version TEXT
        )
        """

    with engine.begin() as conn:
        conn.execute(text(ddl))


def save_prediction(payload: dict, prediction: float | int, model_version: str) -> None:
    engine = get_engine()
    backend = engine.url.get_backend_name()

    init_db()

    payload_value = json.dumps(payload) if backend == "sqlite" else payload

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO predictions(ts, payload, prediction, model_version) "
                "VALUES (:ts, :payload, :pred, :ver)"
            ),
            {
                "ts": dt.datetime.utcnow().isoformat(),
                "payload": payload_value,
                "pred": float(prediction),
                "ver": model_version,
            },
        )
