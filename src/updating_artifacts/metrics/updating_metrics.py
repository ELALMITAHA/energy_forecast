from pathlib import Path
import os
from dotenv import load_dotenv

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from utils.logger import logger
from configs.paths_config import BASE_DIR, METRICS_DIR


class MetricHfUpdate:
    """
    Synchronize model evaluation metrics between local artifacts and Hugging Face.

    This class maintains a persistent metrics history used for monitoring,
    retraining decisions, and auditability in production MLOps pipelines.

    Parameters
    ----------
    model_name : str
        Name of the model whose metrics are updated.

    hf_repo_id : str
        Hugging Face dataset repository ID.

    hf_metrics_dir : str
        Root directory inside the HF repository for storing metrics.

    metrics_file_name : str
        Name of the metrics file (Parquet format).

    Raises
    ------
    EnvironmentError
        If HF_TOKEN is not set in environment variables.

    FileNotFoundError
        If the local run metrics file does not exist.

    RuntimeError
        If any unexpected error occurs during HF metrics download, merge, or upload.

    Notes
    -----
    - Designed for MLOps pipelines: idempotent and reproducible.
    - Logs every step for traceability and monitoring.
    - Works identically in local execution and CI/CD (e.g., GitHub Actions).

    Example
    -------
    >>> updater = MetricHfUpdate(
    >>>     model_name="my_model",
    >>>     hf_repo_id="my-org/my-metrics-dataset",
    >>>     hf_metrics_dir="metrics",
    >>>     metrics_file_name="metrics.parquet"
    >>> )
    >>> updater.update_hf_db()
    """

    # ***** Initialization *****
    def __init__(self, model_name, hf_repo_id, hf_metrics_dir, metrics_file_name):
        self.model_name = model_name
        self.hf_repo_id = hf_repo_id
        self.hf_metrics_dir = hf_metrics_dir
        self.metrics_file_name = metrics_file_name

        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            logger.error("[HF] HF_TOKEN not found in environment variables")
            raise EnvironmentError("HF_TOKEN is required to update HF metrics")

    # ***** Load Metrics from Hugging Face *****
    def _load_hf_metrics_db(self) -> pd.DataFrame:
        """
        Load historical metrics stored on Hugging Face.

        Returns
        -------
        pd.DataFrame
            Historical metrics DataFrame; empty if none exist.
        """
        try:
            subfolder = Path(self.hf_metrics_dir, self.model_name)
            file_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=self.metrics_file_name,
                subfolder=subfolder,
                repo_type="dataset",
                token=self.hf_token,
            )
            df = pd.read_parquet(file_path)
            logger.info(
                f"[HF METRICS] Loaded metrics DB for model={self.model_name} "
                f"(rows={len(df)})"
            )
            return df

        except EntryNotFoundError:
            logger.warning(
                f"[HF METRICS] No existing metrics found for model={self.model_name}. "
                "Initializing empty metrics database."
            )
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"[HF METRICS] Error loading metrics DB: {e}")
            raise RuntimeError(f"Failed to load HF metrics for {self.model_name}") from e

    # ***** Load Metrics from Latest Run *****
    def _load_run_metrics(self) -> pd.DataFrame:
        """
        Load metrics produced by the latest local evaluation run.

        Returns
        -------
        pd.DataFrame
            Metrics DataFrame for the current run.
        """
        run_metrics_path = Path(METRICS_DIR, self.model_name, self.metrics_file_name)
        if not run_metrics_path.exists():
            logger.error(f"[HF METRICS] Local run metrics not found: {run_metrics_path}")
            raise FileNotFoundError(run_metrics_path)

        df = pd.read_parquet(run_metrics_path)
        logger.info(
            f"[HF METRICS] Loaded run metrics for model={self.model_name} "
            f"(rows={len(df)})"
        )
        return df

    # ***** Append Run Metrics to HF Metrics DB *****
    def _append_run_metrics_to_hf_metrics_db(self) -> pd.DataFrame:
        """
        Combine historical HF metrics with latest run metrics.

        Returns
        -------
        pd.DataFrame
            Combined metrics DataFrame.
        """
        df_hf = self._load_hf_metrics_db()
        df_run = self._load_run_metrics()

        if df_run.empty:
            logger.warning("[HF METRICS] Run metrics are empty")

        df_full = pd.concat([df_hf, df_run], ignore_index=True)
        logger.info(f"[HF METRICS] Metrics combined successfully (total_rows={len(df_full)})")
        return df_full

    # ***** Update HF Metrics Database *****
    def update_hf_db(self):
        """
        Execute full metrics update pipeline.

        Combines historical HF metrics with the latest run metrics, saves a temporary
        parquet file, and uploads it to Hugging Face, overwriting the previous version.
        """
        logger.info(f"[HF METRICS] Starting update for model={self.model_name}")

        df_full = self._append_run_metrics_to_hf_metrics_db()

        # ********** Save temporary file **********
        temp_dir = Path(BASE_DIR, "temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / "metrics.parquet"
        df_full.to_parquet(temp_file, index=False)
        logger.info(f"[HF METRICS] Temporary metrics file created: {temp_file}")

        # ********** Upload to HF **********
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=temp_file,
                path_in_repo=f"{self.hf_metrics_dir}/{self.model_name}/{self.metrics_file_name}",
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                token=self.hf_token,
            )
            logger.info(f"[HF METRICS] Metrics successfully updated for model={self.model_name} "
                        f"on repo={self.hf_repo_id}")
        except Exception as e:
            logger.error(f"[HF METRICS] Update failed: {e}")
            raise RuntimeError(f"Failed to upload HF metrics for {self.model_name}") from e
