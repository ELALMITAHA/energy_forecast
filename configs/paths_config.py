from pathlib import Path

# ************ PROJECT ROOT ************
BASE_DIR = Path(__file__).resolve().parents[1]

# ************ LOGS ROOT *******************
LOGS_DIR = BASE_DIR / "logs"

# ************ Models ROOT *******************
MODELS_DIR = BASE_DIR / "models"

# ************ ARTIFACT ROOT (SINGLE SOURCE OF TRUTH) ************
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# ************ SUB ARTIFACTS ************

FORECAST_DIR = ARTIFACTS_DIR / "forecasts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
DATA_QUALITY_DIR = ARTIFACTS_DIR / "data_quality"
DRIFT_DIR = ARTIFACTS_DIR / "drift"
MONITORING_DIR =  ARTIFACTS_DIR / "monitoring"
RETRAIN_FLAG_DIR = MONITORING_DIR / "retrain_flag"

# ************ DATA ************
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_FINAL_DIR = PROCESSED_DIR / "final"

# ************ ENSURE DIRS ************
for d in [
    MODELS_DIR,
    FORECAST_DIR,
    METRICS_DIR,
    DATA_QUALITY_DIR,
    DRIFT_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    PROCESSED_FINAL_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)