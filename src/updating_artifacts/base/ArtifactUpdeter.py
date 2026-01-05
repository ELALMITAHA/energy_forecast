from pathlib import Path
import os

from utils.logger import logger


def overload_artifact(artifact_name, artifact_path, hf_repository):

    artifact_path = Path(artifact_path, artifact_name)

    if not os.path.exists(artifact_path):
        msg = f"[OVERLOAD ARTIFACT] file {artifact_name} not found {artifact_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
