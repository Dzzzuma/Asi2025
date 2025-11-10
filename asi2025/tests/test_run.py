from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def test_kedro_run_smoke(monkeypatch, tmp_path):
    # wyłączamy W&B w testach
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    project_path = Path.cwd()
    bootstrap_project(project_path)

    # Uruchom pipeline i nie rzucaj wyjątków
    with KedroSession.create(project_path=project_path) as session:
        session.run()

    # Sprawdź, że zapisały się metryki baseline
    metrics_path = project_path / "data" / "09_tracking" / "metrics_baseline.json"
    assert metrics_path.exists(), "Brak pliku z metrykami po uruchomieniu pipeline"
