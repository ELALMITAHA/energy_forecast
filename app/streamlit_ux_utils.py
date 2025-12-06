from pathlib import Path
import pandas as pd
import os
import json

def get_data(file_type: str, local_base: str = "data", cloud_base: str = "data-artifact"):
    """
    Charge les données dynamiquement selon l'environnement.
    
    Args:
        file_type (str): "forecast" ou "metrics"
        local_base (str): chemin local racine
        cloud_base (str): chemin cloud/artifact racine

    Returns:
        pd.DataFrame ou dict: dataframe pour forecast, dict pour metrics
    """
    paths = {
        "forecast": ("forecasts/prophet_forecast.parquet", pd.read_parquet),
        "metrics": ("metrics/prophet_metrics.json", lambda f: json.load(open(f))),
    }

    if file_type not in paths:
        raise ValueError(f"file_type doit être 'forecast' ou 'metrics', got '{file_type}'")

    relative_path, loader = paths[file_type]

    # Chemins locaux et cloud
    local_file = Path(local_base) / relative_path
    cloud_file = Path(cloud_base) / relative_path

    # Priorité au local
    if local_file.exists():
        return loader(local_file)
    elif cloud_file.exists():
        return loader(cloud_file)
    else:
        raise FileNotFoundError(f"Aucun fichier trouvé pour {file_type} dans '{local_file}' ni '{cloud_file}'")