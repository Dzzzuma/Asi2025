from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn import metrics

# AutoGluon
try:
    from autogluon.tabular import TabularPredictor
except Exception as e:
    raise ImportError(
        "Brak pakietu 'autogluon'. Dodaj go do environment.yml (pip: autogluon==1.*)."
    ) from e

# Weights & Biases
try:
    import wandb
except Exception:
    wandb = None


def _to_dataframe(
    X: Any, y: Any | None, feature_names: list[str] | None, target_name: str | None
) -> pd.DataFrame:
    """Zamień macierze na DataFrame (z nazwami kolumn) i ewentualnie doklej target."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        X_arr = np.asarray(X)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_arr.shape[1])]
        df = pd.DataFrame(X_arr, columns=feature_names)

    if y is not None:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_series = pd.Series(y).reset_index(drop=True)
        else:
            y_series = pd.Series(np.asarray(y))
        df[(target_name or "target")] = y_series.values
    return df


def _maybe_wandb_init(enable: bool, project: str, job_type: str, config: Dict[str, Any]):
    """Bezpieczne uruchomienie W&B; zwraca kontekst run lub None."""
    if not enable or wandb is None:
        return None
    return wandb.init(project=project, job_type=job_type, config=config)


def _log_wandb(run, payload: Dict[str, Any]):
    if run is not None and wandb is not None:
        wandb.log(payload)


def _save_artifact(run, path: Path, name: str, aliases: list[str] | None = None):
    if run is None or wandb is None:
        return
    art = wandb.Artifact(name=name, type="model")
    art.add_file(str(path))
    wandb.log_artifact(art, aliases=aliases or [])


def train_autogluon(
    X_train,
    y_train,
    feature_names: list[str],
    params: Dict[str, Any],
    wandb_cfg: Dict[str, Any],
    model_pickle_path: str,
):
    """
    Trenuje AutoGluon Tabular i zapisuje predictor jako pickle.

    Inputs:
    - X_train, y_train, feature_names (z katalogu)
    - params: sekcja parameters.autogluon
    - wandb_cfg: sekcja parameters.wandb (enabled, project)
    - model_pickle_path: ścieżka docelowa z catalog.yml (ag_model.filepath)

    Returns:
    - predictor (TabularPredictor)
    """
    target = params.get("label", "target")
    problem_type = params.get("problem_type", None)

    train_df = _to_dataframe(X_train, y_train, feature_names, target)

    run = _maybe_wandb_init(
        enable=bool(wandb_cfg.get("enabled", True)),
        project=wandb_cfg.get("project", "asi2025"),
        job_type="ag-train",
        config=params,
    )

    predictor = TabularPredictor(
        label=target,
        problem_type=problem_type,
        eval_metric=params.get("eval_metric", None),
    )

    predictor.fit(
        train_data=train_df,
        presets=params.get("presets", "medium_quality_faster_train"),
        time_limit=params.get("time_limit", None),
        holdout_frac=params.get("holdout_frac", None),
        verbosity=2,
        random_state=params.get("random_state", 42),
    )

    model_path = Path(model_pickle_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(predictor, f)

    _save_artifact(run, model_path, name="ag_model", aliases=["candidate"])

    if run is not None:
        run.finish()

    return predictor


def evaluate_autogluon(
    predictor,
    X_test,
    y_test,
    feature_names: list[str],
    params: Dict[str, Any],
    wandb_cfg: Dict[str, Any],
    metrics_json_path: str,
) -> Dict[str, float]:
    """
    Ewaluacja modelu AutoGluon + logowanie do W&B i zapis metryk do JSON.

    Returns:
    - dict metryk (np. {"roc_auc": 0.9, "accuracy": 0.8, ...})
    """
    target = params.get("label", "target")
    df_test = _to_dataframe(X_test, None, feature_names, target)

    # Predykcje
    proba = None
    y_pred = predictor.predict(df_test)

    # próbujemy uzyskać prawdopodobieństwa
    try:
        proba = predictor.predict_proba(df_test)
    except Exception:
        proba = None

    y_true = np.asarray(y_test)
    y_hat = np.asarray(y_pred)

    # Wylicz metryki
    results: Dict[str, float] = {}
    problem = params.get("problem_type", "").lower()

    if "regression" in problem:
        results["rmse"] = float(np.sqrt(metrics.mean_squared_error(y_true, y_hat)))
        results["mae"] = float(metrics.mean_absolute_error(y_true, y_hat))
        results["r2"] = float(metrics.r2_score(y_true, y_hat))
    else:
        # klasyfikacja
        results["accuracy"] = float(metrics.accuracy_score(y_true, y_hat))
        # f1 „macro” – działa i dla multi
        results["f1_macro"] = float(metrics.f1_score(y_true, y_hat, average="macro"))
        # ROC AUC dla binary/multi-ovr jeśli mamy proba
        if proba is not None:
            try:
                results["roc_auc_ovr"] = float(
                    metrics.roc_auc_score(y_true, proba, multi_class="ovr")
                )
            except Exception:
                pass

    # Zapis do JSON
    metrics_path = Path(metrics_json_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # W&B
    run = _maybe_wandb_init(
        enable=bool(wandb_cfg.get("enabled", True)),
        project=wandb_cfg.get("project", "asi2025"),
        job_type="ag-eval",
        config=params,
    )
    _log_wandb(run, results)
    if run is not None:
        run.finish()

    return results
