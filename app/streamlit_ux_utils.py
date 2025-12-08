from huggingface_hub import hf_hub_download
import pandas as pd
import json

def get_data(file_type: str, repo_id: str = "tahaelalmi/energy-forecast-artifacts"):
    """
    Charge les données depuis HuggingFace Hub.
    
    Args:
        file_type (str): "forecast" ou "metrics"
        repo_id (str): HuggingFace dataset repo

    Returns:
        pd.DataFrame ou dict
    """
    paths = {
        "forecast": "forecasts/prophet_forecast.parquet",
        "metrics": "metrics/prophet_metrics.json",
    }

    if file_type not in paths:
        raise ValueError(f"file_type doit être 'forecast' ou 'metrics', got '{file_type}'")

    hf_file = paths[file_type]

    # Téléchargement depuis HF Hub
    local_file = hf_hub_download(repo_id=repo_id, filename=hf_file)

    if file_type == "forecast":
        return pd.read_parquet(local_file)
    else:  # metrics
        with open(local_file, "r") as f:
            return json.load(f)