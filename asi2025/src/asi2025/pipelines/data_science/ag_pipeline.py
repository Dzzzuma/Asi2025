from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .ag_nodes import evaluate_autogluon, train_autogluon


def create_pipeline(**kwargs) -> Pipeline:
    """
    Mini-pipeline AutoGluon: trenuj -> ewaluuj -> loguj.

    Zakłada, że wcześniejszy pipeline przygotował:
    X_train, y_train, X_test, y_test, feature_names
    (jak w Twoim catalog.yml).
    """
    return pipeline(
        [
            node(
                func=train_autogluon,
                inputs=dict(
                    X_train="X_train",
                    y_train="y_train",
                    feature_names="feature_names",
                    params="params:autogluon",
                    wandb_cfg="params:wandb",
                    model_pickle_path="params:ag_paths.model_pickle",
                ),
                outputs="ag_predictor",
                name="ag_train",
                tags=["autogluon"],
            ),
            node(
                func=evaluate_autogluon,
                inputs=dict(
                    predictor="ag_predictor",
                    X_test="X_test",
                    y_test="y_test",
                    feature_names="feature_names",
                    params="params:autogluon",
                    wandb_cfg="params:wandb",
                    metrics_json_path="params:ag_paths.metrics_json",
                ),
                outputs="ag_metrics",
                name="ag_eval",
                tags=["autogluon"],
            ),
        ]
    )
