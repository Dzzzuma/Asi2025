from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder

import wandb


# 1) LOAD
def load_raw(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Kedro dostarcza tu DataFrame z catalogu (raw_data)."""
    return raw_data.copy()


# 2) BASIC CLEAN
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimalny, bezpieczny cleaning:
    - usuń duplikaty,
    - usuń kolumny w całości puste,
    - puste ciągi/whitespace → NaN (właściwa imputacja będzie w modelu).
    """
    out = df.copy()
    out = out.drop_duplicates()
    out = out.dropna(axis=1, how="all")
    out = out.replace(r"^\s*$", pd.NA, regex=True)
    return out


# 3) SPLIT
def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dzielimy na X/y oraz train/test.
    Zwracamy y jako DataFrame (nie Series), żeby ParquetDataset mógł to zapisać.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Kolumna celu '{target_col}' nie istnieje. " f"Dostępne: {list(df.columns)[:20]}..."
        )

    y = df[target_col]
    X = df.drop(columns=[target_col])

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=strat,
    )

    # <- KLUCZOWE: ParquetDataset oczekuje DataFrame, nie Series
    y_train = y_train.to_frame(name=target_col)
    y_test = y_test.to_frame(name=target_col)

    return X_train, X_test, y_train, y_test


# 4) TRAIN (pełny preprocessing w środku modelu)
def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,  # DataFrame z jedną kolumną (target)
    params: Dict[str, Any] | None,
):
    """
    Sklearn Pipeline:
      - num: imputacja medianą
      - cat: imputacja najczęstszą + OneHotEncoder(handle_unknown="ignore")
      - model: LogisticRegression(max_iter z params)
    Dzięki temu pipeline poradzi sobie z kategorycznymi ('Female') i NaN.
    """



    import wandb

    # --- LOGOWANIE DO W&B ---
    wandb.init(
        project="asi2025",  # 🔹 nazwa projektu w wandb.ai
        job_type="train",
        config=params or {},
        settings=wandb.Settings(start_method="thread"),  # bezpieczne dla Kedro
    )

    # --- PARAMETRY MODELU ---
    params = params or {}
    max_iter = int(params.get("max_iter", 5000))

    # --- PRZYGOTOWANIE KOLUMN ---
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    clf = SkPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=max_iter)),
        ]
    )

    # --- TRENING ---
    y_train_1d = y_train.iloc[:, 0]
    clf.fit(X_train, y_train_1d)

    # --- LOGOWANIE INFORMACJI O TRENINGU ---
    wandb.log({"train_samples": len(X_train)})

    # --- ZAPISZ MODEL LOKALNIE ---
    output_path = Path("data/06_models/model_baseline.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)

    # --- WYŚLIJ MODEL JAKO ARTEFAKT DO W&B ---
    artifact = wandb.Artifact(
        name="model_baseline",
        type="model",
        description="Baseline model trained with LogisticRegression",
    )
    artifact.add_file(str(output_path))
    wandb.log_artifact(artifact)

    # --- ZAKOŃCZ RUN W&B ---
    wandb.finish()

    return clf


# 5) EVALUATE
def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Dict[str, float]:
    """
    Zwraca metryki jako dict (zapisze je JSONDataset).
    - accuracy (zawsze),
    - f1_weighted (zawsze),
    - roc_auc (gdy binary i model ma predict_proba).
    """

    # jeśli nie ma aktywnego runa, inicjuj nowy
    if wandb.run is None:
        wandb.init(project="asi2025", job_type="evaluate", reinit=True)

    metrics: Dict[str, float] = {}

    y_true = y_test.iloc[:, 0]
    y_pred = model.predict(X_test)

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))

    # ROC AUC dla klasyfikacji binarnej (jeśli dostępne proby)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
    except Exception:
        pass

    # --- LOGOWANIE METRYK DO W&B ---
    wandb.log(metrics)
    wandb.finish()

    return metrics
