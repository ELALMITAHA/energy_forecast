import os 
from huggingface_hub import hf_hub_download
import pandas as pd
import json
from pathlib import Path

def get_data(file_type: str, repo_id: str = "tahaelalmi/energy-forecast-artifacts"):
    """
    Charge toujours les données depuis HuggingFace Hub (repo public).

    Args:
        file_type (str): "forecast", "metrics", "data_quality", "retrain_flag"
        repo_id (str): HuggingFace dataset repo

    Returns:
        pd.DataFrame ou dict
    """
    paths = {
        "forecast": ("forecasts/prophet/forecasts.parquet", pd.read_parquet),
        "metrics": ("metrics/prophet/metrics.parquet", pd.read_parquet),
        "data_quality": ("data_quality/data_quality_report.json", lambda f: json.load(open(f))),
        "retrain_flag": ("monitoring/retrain_flag/retrain_flag.json", lambda f: json.load(open(f))),
    }

    if file_type not in paths:
        raise ValueError(f"file_type doit être un de {list(paths.keys())}, got '{file_type}'")

    relative_path, loader = paths[file_type]
    rel_path_obj = Path(relative_path)
    subfolder = str(rel_path_obj.parent)
    filename = rel_path_obj.name

    hf_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        repo_type="dataset",  # repo private
        token=os.getenv("HF_TOKEN") 
    )

    return loader(hf_file)
