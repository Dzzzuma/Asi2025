import wandb
from pathlib import Path


MODEL_PATH = Path("data/06_models/ag_production.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Nie znaleziono pliku modelu: {MODEL_PATH}")

run = wandb.init(
    project="asi2025",                    
    job_type="register-production-model",  
)

artifact = wandb.Artifact(
    name="ag_model",                       
    type="model",                          
    description=(
        "AutoGluon production model wybrany na podstawie "
        "najwyższego ROC AUC na zbiorze testowym."
    ),
)


artifact.add_file(str(MODEL_PATH))


wandb.log_artifact(artifact, aliases=["candidate", "production"])

run.finish()
print("Zalogowano artefakt 'ag_model' z aliasami: candidate, production")
