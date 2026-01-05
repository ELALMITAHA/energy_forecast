import os
from pathlib import Path
import json

import pandas as pd

from utils.logger import logger


def overload_artifact(artifact_name, artifact_path, hf_repository):

    artifact_path = Path(artifact_path, artifact_name)

    if not os.path.exists(artifact_path):
        msg = f"[OVERLOAD ARTIFACT] file {artifact_name} not found {artifact_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if artifact_name.endswith(".json"):
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif artifact_name.endswith(".parquet"):
        data = pd.read_parquet(artifact_path)
    else:
        msg = (
            f"[OVERLOAD ARTIFACT] Unsupported artifact extension | "
            f"artifact={artifact_name} | path={artifact_path}"
        )
        logger.error(msg)
        raise ValueError(msg)
