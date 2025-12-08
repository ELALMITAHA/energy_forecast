from huggingface_hub import hf_hub_download
import pandas as pd
import json
from pathlib import Path


def get_data(file_type: str,
             local_base: str = "data",
             repo_id: str = "tahaelalmi/energy-forecast-artifacts"):
    """
    Charge les données dynamiquement selon l'environnement :
    - local si le fichier existe,
    - sinon depuis HuggingFace Hub.

    Args:
        file_type (str): "forecast" ou "metrics"
        local_base (str): chemin racine local pour fallback
        repo_id (str): HuggingFace dataset repo

    Returns:
        pd.DataFrame ou dict
    """
    paths = {
        "forecast": ("forecasts/prophet_forecast.parquet", pd.read_parquet),
        "metrics": ("metrics/prophet_metrics.json", lambda f: json.load(open(f))),
    }

    if file_type not in paths:
        raise ValueError(f"file_type doit être 'forecast' ou 'metrics', got '{file_type}'")

    relative_path, loader = paths[file_type]
    local_file = Path(local_base) / relative_path

    # Lecture locale si disponible
    if local_file.exists():
        return loader(local_file)

    # Sinon téléchargement depuis HuggingFace Hub
    hf_file = hf_hub_download(repo_id=repo_id,
                              filename=relative_path,
                              repo_type="dataset")  # public dataset, pas de token nécessaire
    return loader(hf_file)

if __name__ =="__main__":

    # Test
    df_forecast = get_data("forecast")
    metrics = get_data("metrics")

    print(df_forecast.tail())
    print(metrics)