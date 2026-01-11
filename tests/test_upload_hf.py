from pathlib import Path
from utils.upload_hf import upload_to_hf_repo
from configs.paths_config import (
    DATA_QUALITY_DIR,
    FORECAST_DIR,
    RETRAIN_FLAG_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR
    )

DATASET_ID = "tahaelalmi/energy-forecast-artifacts"
MODEL_REPO_ID = "tahaelalmi/energy-forecast-models"

upload_to_hf_repo(
    local_folder_path=DATA_QUALITY_DIR,
    hf_repo_id="tahaelalmi/energy-forecast-artifacts",
    hf_repo_path=str(DATA_QUALITY_DIR.relative_to(ARTIFACTS_DIR)),  # <-- convert to str
    hf_repo_type="dataset"  
)

upload_to_hf_repo(
    local_folder_path=FORECAST_DIR / "prophet",
    hf_repo_id="tahaelalmi/energy-forecast-artifacts",
    hf_repo_path=str((FORECAST_DIR / "prophet").relative_to(ARTIFACTS_DIR)),  # <-- convert to str
    hf_repo_type="dataset"
)

upload_to_hf_repo(
    local_folder_path=RETRAIN_FLAG_DIR,
    hf_repo_id="tahaelalmi/energy-forecast-artifacts",
    hf_repo_path=str(RETRAIN_FLAG_DIR.relative_to(ARTIFACTS_DIR)),  # <-- convert to str
    hf_repo_type="dataset")

upload_to_hf_repo(
    local_folder_path=MODELS_DIR,
    hf_repo_id=MODEL_REPO_ID,
    hf_repo_path=str((MODELS_DIR / "prophet").relative_to(MODELS_DIR)),
    hf_repo_type="model"
)
