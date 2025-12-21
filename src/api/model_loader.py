import os
from pathlib import Path

import joblib


def ensure_model_downloaded() -> str:
    """
    Pobiera model z W&B artifactu do MODEL_PATH jeśli go nie ma.
    Wymaga env:
      WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, WANDB_ARTIFACT
    Zwraca ścieżkę do modelu.
    """
    model_path = os.getenv("MODEL_PATH", "/app/model/model.pkl")
    p = Path(model_path)

    if p.exists() and p.stat().st_size > 0:
        return str(p)

    import wandb

    api_key = os.getenv("WANDB_API_KEY")
    entity = os.getenv("WANDB_ENTITY", "j-lukassiak-pjatk")
    project = os.getenv("WANDB_PROJECT", "asi2025")
    artifact = os.getenv("WANDB_ARTIFACT")  # np. "ag_model:production"

    if not api_key:
        raise RuntimeError("Brakuje WANDB_API_KEY (sekret w Cloud Run)")
    if not artifact:
        raise RuntimeError("Brakuje WANDB_ARTIFACT np. ag_model:production")

    wandb.login(key=api_key, relogin=True)

    run = wandb.init(
        project=project,
        entity=entity,
        job_type="download-model",
        reinit=True,
    )

    art = run.use_artifact(f"{entity}/{project}/{artifact}")
    art_dir = Path(art.download(root="/app/model_artifact"))
    run.finish()

    candidates = [x for x in art_dir.rglob("*") if x.is_file() and x.suffix in (".pkl", ".joblib")]

    if not candidates:
        raise RuntimeError(f"Nie znaleziono pliku .pkl/.joblib w artefakcie: {art_dir}")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(candidates[0].read_bytes())

    return str(p)


def load_model():
    path = ensure_model_downloaded()
    return joblib.load(path)
