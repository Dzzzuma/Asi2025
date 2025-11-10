from kedro.pipeline import Pipeline, node, pipeline

from .nodes import basic_clean, evaluate, load_raw, split_data, train_baseline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=load_raw, inputs="raw_data", outputs="raw_loaded", name="load"),
            node(func=basic_clean, inputs="raw_loaded", outputs="clean_data", name="clean"),
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
        ]
    )
