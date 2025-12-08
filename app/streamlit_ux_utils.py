from huggingface_hub import hf_hub_download
import pandas as pd
import json

def get_data(file_type: str, repo_id: str = "tahaelalmi/energy-forecast-artifacts"):
    """
    Charge un artifact (forecast ou metrics) depuis HuggingFace Hub.

    Args:
        file_type (str): "forecast" ou "metrics"
        repo_id (str): nom du dataset HuggingFace

    Returns:
        pd.DataFrame pour forecast
        dict pour metrics
    """

    files = {
        "forecast": "forecasts/prophet_forecast.parquet",
        "metrics": "metrics/prophet_metrics.json",
    }

    if file_type not in files:
        raise ValueError(f"[get_data] invalid file_type='{file_type}'. Use: forecast | metrics")

    # Download depuis HF
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=files[file_type],
        repo_type="dataset"
    )

    # Parsing
    if file_type == "forecast":
        return pd.read_parquet(local_path)

    with open(local_path, "r") as f:
        return json.load(f)
