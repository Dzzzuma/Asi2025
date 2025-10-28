from kedro.pipeline import Pipeline, pipeline
from asi2025.pipelines.data_science.pipeline import create_pipeline as ds_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "data_science": ds_pipeline(),
        "__default__": ds_pipeline(),
    }
