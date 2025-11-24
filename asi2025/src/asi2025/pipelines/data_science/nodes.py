from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1) wczytanie danych
def load_raw(df: pd.DataFrame, sample_n: int | None = None) -> pd.DataFrame:
    """Zwróć surowy dataframe (opcjonalnie próbkę n wierszy)."""
    if sample_n is not None and sample_n > 0 and sample_n < len(df):
        return df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    return df


# 2) proste czyszczenie
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimalne czyszczenie: usuwanie duplikatów, strip spacji w stringach."""
    df = df.copy()

    # wyczyść białe znaki w kolumnach tekstowych
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()

    # usuń duplikaty
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


# 3) podział na zbiory + przygotowanie cech
def train_test_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, list[str]]:
    """Zwraca X_train, X_test, y_train, y_test oraz listę nazw cech po preprocessingu."""
    assert target_col in df.columns, f"Nie znaleziono kolumny celu: {target_col}"

    y = df[target_col]
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    try:
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    except TypeError:
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

    preprocess = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])

    X_train_ready = preprocess.fit_transform(X_train)
    X_test_ready = preprocess.transform(X_test)

    feature_names: list[str] = list(num_cols)
    ohe = preprocess.named_transformers_["cat"].named_steps["ohe"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names += cat_feature_names

    # zwracamy gotowe macierze + labelle bez mapowania
    return (
        X_train_ready,
        X_test_ready,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        feature_names,
    )


# 4) trenowanie modelu baseline
def train_baseline(X_train: np.ndarray, y_train: pd.Series, params: dict):
    """
    Trenuje bazowy model (LogisticRegression + StandarScaler).
    params: słownik z kluczami np. {"max_iter": 2000, "C": 1.0}
    """
    # mapowanie etykiet na 0/1
    label_map = {"Satisfied": 1, "Neutral or Dissatisfied": 0}
    y_train_bin = y_train.map(label_map).astype("int8")

    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            max_iter=int(params.get("max_iter", 2000)),
            C=float(params.get("C", 1.0)),
            random_state=int(params.get("random_state", 42)),
        ),
    )
    model.fit(X_train, y_train_bin)
    return model


# 5) ewaluacja
def evaluate(model, X_test: np.ndarray, y_test: pd.Series) -> dict[str, float]:
    """Zwraca słownik metryk: roc_auc i f1."""
    label_map = {"Satisfied": 1, "Neutral or Dissatisfied": 0}
    y_test_bin = y_test.map(label_map).astype("int8")

    y_pred = model.predict(X_test)
    # jeśli jest metoda predict_proba – policz ROC AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test_bin, y_proba))
    else:
        roc_auc = float("nan")

    f1 = float(f1_score(y_test_bin, y_pred))

    return {"roc_auc": roc_auc, "f1": f1}


# 6) trenowanie modelu AutoGluon
def train_autogluon(
    X_train: np.ndarray,
    y_train: pd.Series,
    feature_names: list[str],
    params: dict,
) -> TabularPredictor:

    label_col = params.get("label", "Satisfaction")

    df_train = pd.DataFrame(X_train, columns=feature_names).copy()
    df_train[label_col] = y_train.values

    predictor = TabularPredictor(
        label=label_col,
        problem_type=params.get("problem_type"),
        eval_metric=params.get("eval_metric"),
    ).fit(
        train_data=df_train,
        presets=params.get("presets", "medium_quality_faster_train"),
        time_limit=int(params.get("time_limit", 120)),
        seed=int(params.get("seed", 42)),
    )

    return predictor


# 7) ewaluacja AutoGluon
def evaluate_autogluon(
    ag_predictor: TabularPredictor,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: list[str],
    params: dict,
) -> dict[str, float]:

    label_col = params.get("label", "Satisfaction")

    df_test = pd.DataFrame(X_test, columns=feature_names).copy()
    df_test[label_col] = y_test.values

    eval_metrics = ag_predictor.evaluate(df_test, auxiliary_metrics=True)

    metrics: dict[str, float] = {}
    for k, v in eval_metrics.items():
        try:
            metrics[k] = float(v)
        except (TypeError, ValueError):
            continue

    return metrics
