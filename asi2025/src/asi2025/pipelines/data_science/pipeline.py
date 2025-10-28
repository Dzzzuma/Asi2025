from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import basic_clean, evaluate, load_raw, train_baseline, train_test_split


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            # 1) load
            node(
                func=load_raw,
                inputs=["raw_airline", "params:sample_n"],
                outputs="raw_df",
                name="load_raw_node",
            ),
            # 2) clean
            node(
                func=basic_clean,
                inputs="raw_df",
                outputs="clean_df",
                name="basic_clean_node",
            ),
            # 3) split
            node(
                func=train_test_split,
                inputs=dict(
                    df="clean_df",
                    target_col="params:target_col",
                    test_size="params:test_size",
                    random_state="params:random_state",
                ),
                outputs=["X_train", "X_test", "y_train", "y_test", "feature_names"],
                name="split_node",
            ),
            # 4) train_baseline
            node(
                func=train_baseline,
                inputs=["X_train", "y_train", "params:model"],
                outputs="model_baseline",
                name="train_baseline_node",
            ),
            # 5) evaluate
            node(
                func=evaluate,
                inputs=["model_baseline", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_node",
            ),
        ]
    )
