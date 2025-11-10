from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    basic_clean,
    evaluate,
    evaluate_autogluon,
    load_raw,
    split_data,
    train_autogluon,
    train_baseline,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # 1) load
            node(
                func=load_raw,
                inputs="raw_data",
                outputs="raw_loaded",
                name="load",
            ),
            # 2) basic clean
            node(
                func=basic_clean,
                inputs="raw_loaded",
                outputs="clean_data",
                name="clean",
            ),
            # 3) split
            node(
                func=split_data,
                inputs=[
                    "clean_data",
                    "params:target_col",
                    "params:split.test_size",
                    "params:split.random_state",
                    "params:split.stratify",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split",
            ),
            # 4) baseline
            node(
                func=train_baseline,
                inputs=["X_train", "y_train", "params:model"],
                outputs="model_baseline",
                name="train_baseline",
            ),
            node(
                func=evaluate,
                inputs=["model_baseline", "X_test", "y_test"],
                outputs="metrics_baseline",
                name="evaluate",
            ),
            # 5) AutoGluon – trening
            node(
                func=train_autogluon,
                inputs=["X_train", "y_train", "params:autogluon"],
                outputs="ag_model",
                name="train_autogluon",
            ),
            # 6) AutoGluon – ewaluacja
            node(
                func=evaluate_autogluon,
                inputs=["ag_model", "X_test", "y_test", "params:autogluon"],
                outputs="ag_metrics",
                name="evaluate_autogluon",
            ),
        ]
    )
