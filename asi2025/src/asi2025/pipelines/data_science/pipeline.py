"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node

from .nodes import add_one


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=add_one,
                inputs="params:seed",
                outputs="result",
                name="add_one_node",
            )
        ]
    )
