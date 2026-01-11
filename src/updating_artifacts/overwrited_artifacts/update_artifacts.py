import os
from dotenv import load_dotenv
from pathlib import Path
from utils.upload_hf import upload_to_hf_repo
from huggingface_hub import HfApi

from configs.paths_config import (
    DATA_QUALITY_DIR,
    FORECAST_DIR,
    RETRAIN_FLAG_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR,
    LOGS_DIR
)

# ***** Hugging Face repository IDs *****
DATASET_ID = "tahaelalmi/energy-forecast-artifacts"
MODEL_REPO_ID = "tahaelalmi/energy-forecast-models"

# ***** Upload Data Quality Reports *****
upload_to_hf_repo(
    local_folder_path=DATA_QUALITY_DIR,
    hf_repo_id=DATASET_ID,
    hf_repo_path=str(DATA_QUALITY_DIR.relative_to(ARTIFACTS_DIR)),
    hf_repo_type="dataset"
)

# ***** Upload Forecasts *****
upload_to_hf_repo(
    local_folder_path=FORECAST_DIR / "prophet",
    hf_repo_id=DATASET_ID,
    hf_repo_path=str((FORECAST_DIR / "prophet").relative_to(ARTIFACTS_DIR)),
    hf_repo_type="dataset"
)

# ***** Upload Retrain Flags *****
upload_to_hf_repo(
    local_folder_path=RETRAIN_FLAG_DIR,
    hf_repo_id=DATASET_ID,
    hf_repo_path=str(RETRAIN_FLAG_DIR.relative_to(ARTIFACTS_DIR)),
    hf_repo_type="dataset"
)

# ***** Upload Models *****
upload_to_hf_repo(
    local_folder_path=MODELS_DIR,
    hf_repo_id=MODEL_REPO_ID,
    hf_repo_path=str((MODELS_DIR / "prophet").relative_to(MODELS_DIR)),
    hf_repo_type="model"
)

# ***** Upload Log File *****
LOG_FILE = Path("logs/energy_forecast.log")
HF_LOG_NAME = "run_log_last_run.log"  # file at the root of HF dataset

load_dotenv()
token = os.getenv("HF_TOKEN")

api = HfApi()

api.upload_file(
    path_or_fileobj=LOG_FILE,
    path_in_repo=HF_LOG_NAME,
    repo_id=DATASET_ID,
    repo_type="dataset",
    token=token
)
