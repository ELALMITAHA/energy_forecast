import os
from pathlib import Path
from dotenv import load_dotenv

from huggingface_hub import HfApi

from utils.logger import logger
from configs.paths_config import BASE_DIR


def upload_to_hf_repo(
        local_folder_path,
        hf_repo_id,
        hf_repo_type,
        hf_repo_path,
        ):
    """
    Upload a local folder to an existing Hugging Face repository.

    This function uploads all files contained in a local directory to a
    Hugging Face Hub repository using the Hugging Face API. The repository
    must already exist on the Hub. Authentication is handled via the
    `HF_TOKEN` environment variable.

    Parameters
    ----------
    local_folder_path : str or pathlib.Path
        Path to the local folder containing artifacts to upload.
        Can be absolute or relative to the project root.
    hf_repo_id : str
        Hugging Face repository identifier, including namespace.
        Example: "username/repository-name" or "org-name/repository-name".
    hf_repo_type : str
        Type of the Hugging Face repository. Must be one of: "model", "dataset", or "space".
    hf_repo_path : str
        Path inside the HF repository where the folder should be uploaded.
        Example: "artifacts/data_quality".

    Raises
    ------
    ValueError
        If the Hugging Face token is not found in environment variables
        or if the local folder path does not exist.
    Exception
        Propagates any exception raised by the Hugging Face Hub API
        during the upload process.

    Notes
    -----
    - The function assumes the target repository already exists on Hugging Face.
    - Hugging Face token is never logged for security reasons.
    - Designed for MLOps pipelines: idempotent, fault-tolerant, fully logged.

    Example
    -------
    >>> from utils.upload_hf import upload_to_hf_repo
    >>> from configs.paths_config import DATA_QUALITY_DIR
    >>> upload_to_hf_repo(
    ...     local_folder_path=DATA_QUALITY_DIR,
    ...     hf_repo_id="tahaelalmi/energy-forecast-artifacts",
    ...     hf_repo_type="dataset",
    ...     hf_repo_path="data_quality"
    ... )
    """
    # ***** Load environment variables *****
    load_dotenv()

    # ***** Resolve Hugging Face token *****
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        logger.error(
            "[HF UPLOAD] HF_TOKEN not found. "
            "No token provided as argument and no HF_TOKEN in environment variables."
        )
        raise ValueError(
            "HF_TOKEN must be provided or set in environment variables"
        )

    # ***** Build and validate local folder path *****
    local_folder_path = Path(local_folder_path)
    if not local_folder_path.exists():
        logger.error(
            f"[HF UPLOAD] {local_folder_path} not found in "
            f"{local_folder_path.relative_to(BASE_DIR)}"
        )
        raise ValueError(f"{local_folder_path} does not exist")

    try:
        # ***** Initialize Hugging Face API client *****
        api = HfApi()

        # ***** Upload folder to Hugging Face repository *****
        api.upload_folder(
            repo_id=hf_repo_id,
            folder_path=local_folder_path,
            repo_type=hf_repo_type,
            path_in_repo=hf_repo_path,
            token=hf_token
        )

    except Exception as e:
        logger.error(
            f"[HF UPLOAD] uploading {local_folder_path} to {hf_repo_id} failed"
        )
        raise




