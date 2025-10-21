"""Project pipelines."""

from __future__ import annotations

from asi2025.pipelines.data_science import pipeline as ds


def register_pipelines():
    ds_pipe = ds.create_pipeline()
    return {
        "data_science": ds_pipe,
        "__default__": ds_pipe,
    }
