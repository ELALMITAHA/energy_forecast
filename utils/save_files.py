import json
from pathlib import Path
from datetime import datetime

import pandas as pd 

from configs.paths_config import BASE_DIR
from utils.logger import logger


# ***** Save Data Validation Report *****
def save_validation_report(report, output_path, filename="data_quality_report.json"):
    """
    Save a data quality validation report as a JSON artifact.

    Parameters
    ----------
    report : dict
        Data quality report returned by the validator.
    output_path : str or Path
        Directory where the report will be saved.
    filename : str, optional
        Name of the JSON artifact (default="data_quality_report.json").

    Returns
    -------
    Path
        Path to the saved JSON report.
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define full path to report file
    file_path = output_path / filename

    # Write JSON file
    with open(file_path, "w") as f:
        json.dump(report, f, indent=2, default=int)

    # Log success
    logger.info(f"[DATA VALIDATION] Report saved to {file_path.relative_to(BASE_DIR)}")

    return file_path


# ***** Save Model Metrics (overwrite) *****
def save_metrics(
    metrics: dict,
    output_dir: Path,
    model_name: str,
):
    """
    Save model evaluation metrics as a JSON artifact, overwriting previous metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics.
    output_dir : Path or str
        Base directory to save metrics for the model.
    model_name : str
        Name of the model (used for organizing output).

    Returns
    -------
    Path
        Path to the saved metrics JSON file.
    """

    # Combine metrics with timestamp
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    metrics_payload = {
        "timestamp_utc": timestamp,
        **metrics,
    }

    df = pd.DataFrame([metrics_payload])

    # Define file path (always same name, overwrites)
    save_path = Path(output_dir,model_name,"metrics.parquet")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False)

    logger.info(
        f"[METRICS] Consumption data saved to {save_path.relative_to(BASE_DIR)}"
    )

    # Log success
    logger.info(f"[METRICS] Metrics saved to {save_path.relative_to(BASE_DIR)}")

