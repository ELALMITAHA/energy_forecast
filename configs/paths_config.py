from pathlib import Path
import os

# ************ PROJECT ROOT ************
BASE_DIR = Path(__file__).resolve().parent.parent  # toujours energy_forecast/

# ************ DATA DIRECTORIES ************
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_FINAL_DIR = PROCESSED_DIR / "final"

# ************ ARTIFACTS DIRECTORIES ************
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_QUALITY_DIR = ARTIFACTS_DIR / "data_quality"
MODELS_DIR = ARTIFACTS_DIR / "models"
FORECAST_DIR = ARTIFACTS_DIR / "forecasts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# ************ ENVIRONMENT DETECTION ************
IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"
if IS_GITHUB_ACTIONS:
    BASE_TMP = Path("/tmp/artifacts")
    RAW_DIR = BASE_TMP / "data/raw"
    PROCESSED_DIR = BASE_TMP / "data/processed"
    PROCESSED_FINAL_DIR = BASE_TMP / "data/processed/final"
    FORECAST_DIR = BASE_TMP / "data/forecasts"
    METRICS_DIR = BASE_TMP / "data/metrics"
    MODELS_DIR = BASE_TMP / "models"
    DATA_QUALITY_DIR = BASE_TMP / "data_quality"

# ************ ENSURE DIRECTORIES EXIST ************
for d in [
    RAW_DIR,
    PROCESSED_DIR,
    PROCESSED_FINAL_DIR,
    FORECAST_DIR,
    METRICS_DIR,
    MODELS_DIR,
    DATA_QUALITY_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
