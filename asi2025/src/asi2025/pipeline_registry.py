from __future__ import annotations

from kedro.pipeline import Pipeline

# Nowy pipeline AutoGluon (Sprint 3 – zadanie 3)
# Twój dotychczasowy pipeline (Sprint 1–2)
from asi2025.pipelines.data_science import ag_pipeline
from asi2025.pipelines.data_science import pipeline as ds_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """
    Rejestruje potoki Kedro.

    - "data_science": Twój istniejący pipeline (load->clean->split->baseline itp.)
    - "autogluon":    Mini pipeline z treningiem i ewaluacją AG
    - "__default__":  Połączenie obu, więc `kedro run` zrobi jedno i drugie.
    """
    data_science = ds_pipeline.create_pipeline()
    autogluon = ag_pipeline.create_pipeline()

    return {
        "data_science": data_science,
        "autogluon": autogluon,
        "__default__": data_science + autogluon,
    }
