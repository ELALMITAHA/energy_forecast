from pathlib import Path
import os


# --- Racine du projet ---
BASE_DIR = Path(__file__).resolve().parents[1]  # racine du projet

# --- Dossier pour les models ---
MODELS_DIR = Path(BASE_DIR,"models")

# --- Dossiers datas ---
DATA_DIR = BASE_DIR / "data"
# --- Dossier sauvegarde données brutes 
RAW_DIR = Path(DATA_DIR , "raw")
# --- Dossier sauvegarde données traitées
PROCESSED_DIR = Path(DATA_DIR , "processed")
# --- Dossier sauvegarde prédictions
FORECAST_DIR = Path( DATA_DIR , "forecasts")
# Sauvegarder les metrics 
METRICS_DIR = Path(DATA_DIR,"metrics")

# --- Détecter l'environnement ---
IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"

if IS_GITHUB_ACTIONS:
    BASE_TMP = Path("/tmp/artifacts")
    RAW_DIR = BASE_TMP / "data/raw"
    PROCESSED_DIR = BASE_TMP / "data/processed"
    FORECAST_DIR = BASE_TMP / "data/forecasts"
    METRICS_DIR = BASE_TMP / "data/metrics"
    MODELS_DIR = BASE_TMP / "models"

for d in [RAW_DIR, PROCESSED_DIR, FORECAST_DIR, METRICS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


