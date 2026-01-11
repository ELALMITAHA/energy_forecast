import json
from pathlib import Path
import os
from dotenv import load_dotenv

from utils.logger import logger
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


def load_retrain_flag(hf_repo_id, retrain_flag_file_name="retrain_flag.json"):
    """
    Load retraining flag from HF dataset.  

    Returns True if:
    - The flag file says retrain, OR
    - The flag file cannot be found or read (safe default)

    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    try:
        # ⚠ filename = juste le nom, subfolder = dossier
        file_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=retrain_flag_file_name,
            subfolder=str(Path("monitoring") / "retrain_flag"),
            repo_type="dataset",
            token=hf_token,
            force_download=True,
        )
        logger.info(f"[RETRAIN FLAG] Flag file downloaded: {file_path}")

    except EntryNotFoundError:
        logger.warning(f"[RETRAIN FLAG] File not found in HF dataset '{hf_repo_id}'. Defaulting to should_retrain=True")
        return True
    except Exception as e:
        logger.error(f"[RETRAIN FLAG] Error downloading flag: {e}. Defaulting to should_retrain=True")
        return True

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        should_retrain = data.get("should_retrain", True)
        logger.info(f"[RETRAIN FLAG] Flag read successfully: should_retrain={should_retrain}")
    except Exception as e:
        logger.error(f"[RETRAIN FLAG] Error reading flag file: {e}. Defaulting to should_retrain=True")
        should_retrain = True

    return should_retrain


