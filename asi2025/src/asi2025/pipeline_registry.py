"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from asi2025.pipelines.data_science.pipeline import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    data_science = create_pipeline()
    return {"__default__": data_science, "data_science": data_science}
