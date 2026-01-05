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

# ************ ENVIRONMENT DETECTION & PATHS ************
RAW_DIR = Path(os.environ.get("RAW_DIR", "./tmp/data/raw"))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", "./tmp/data/processed"))
PROCESSED_FINAL_DIR = Path(os.environ.get("PROCESSED_FINAL_DIR", "./tmp/data/processed/final"))
FORECAST_DIR = Path(os.environ.get("FORECAST_DIR", "./tmp/artifacts/forecasts"))
METRICS_DIR = Path(os.environ.get("METRICS_DIR", "./tmp/artifacts/metrics"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./tmp/artifacts/models"))
DATA_QUALITY_DIR = Path(os.environ.get("DATA_QUALITY_DIR", "./tmp/artifacts/data_quality"))

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
