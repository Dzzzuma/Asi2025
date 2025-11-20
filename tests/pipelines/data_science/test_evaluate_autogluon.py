import sys
from pathlib import Path

import numpy as np
import pandas as pd

# import pakietu z katalogu src
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from asi2025.pipelines.data_science.nodes import evaluate_autogluon


class DummyPredictor:
    def predict(self, X: pd.DataFrame):
        # zawsze zwraca same zera
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame):
        # stałe prawdopodobieństwa 0.5 / 0.5
        proba_0 = np.full(len(X), 0.5)
        proba_1 = np.full(len(X), 0.5)
        return pd.DataFrame({0: proba_0, 1: proba_1})

    def evaluate(self, *args, **kwargs):
        # zwraca słownik metryk w [0, 1]
        return {
            "roc_auc": 0.7,
            "accuracy": 0.8,
            "f1_weighted": 0.75,
        }


def test_evaluate_autogluon_returns_valid_metrics():
    X_test = pd.DataFrame({"feature1": [0.1, 0.2, 0.8, 0.9]})
    # y_test jako DataFrame, bo evaluate_autogluon używa y_test.iloc[:, 0]
    y_test = pd.DataFrame({"target": [0, 0, 1, 1]})

    dummy_predictor = DummyPredictor()
    params = {
        "eval_metric": "roc_auc",
        "label": "target",
    }

    metrics = evaluate_autogluon(dummy_predictor, X_test, y_test, params)

    assert isinstance(metrics, dict)

    expected_keys = {"roc_auc", "accuracy", "f1_weighted"}

    for key in expected_keys:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_models_directory_exists():
    models_dir = Path("data/06_models")
    assert models_dir.exists() and models_dir.is_dir()


def test_production_model_file_exists():
    model_path = Path("data/06_models/ag_production.pkl")
    assert model_path.exists()
