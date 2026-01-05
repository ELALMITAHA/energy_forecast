import json
from pathlib import Path
from datetime import datetime
import uuid

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


# ***** Save Model Metrics *****
def save_metrics(
    metrics: dict,
    output_dir: Path,
    model_name: str,
    model_version: str,
):
    """
    Save model evaluation metrics as a JSON artifact.

    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics.
    output_dir : Path or str
        Base directory to save metrics for the model.
    model_name : str
        Name of the model (used for organizing output).
    model_version : str
        Version of the model (used for reporting).

    Returns
    -------
    Path
        Path to the saved metrics JSON file.
    """
    # Create model-specific output directory
    output_dir = Path(output_dir, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique run ID and timestamp
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.utcnow().isoformat(timespec="seconds")

    # Combine metrics with metadata
    metrics_payload = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "model_name": model_name,
        "model_version": model_version,
        **metrics,
    }

    # Define file path
    file_path = output_dir / f"{timestamp}_run_{run_id}.json"

    # Save metrics to JSON
    with open(file_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Log success
    logger.info(f"[METRICS] Metrics saved to {file_path.relative_to(BASE_DIR)}")

    return file_path
