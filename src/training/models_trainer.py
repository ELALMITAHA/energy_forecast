import importlib
from pathlib import Path
import yaml
import joblib

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet

from configs.paths import PROCESSED_DIR,MODELS_DIR, FORECAST_DIR, MODELS_DIR, METRICS_DIR

MODELS_CONFIG_DIR = Path("configs")
MODELS_CONFIG = {}

# ---- Lire toutes les configurations des models ----
for file in MODELS_CONFIG_DIR.glob("*.yaml"):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
        # utiliser le nom du fichier (sans .yaml) comme clé
        key = file.stem
        MODELS_CONFIG[key] = config

def train_and_forecast(model_name : str):

    # ---- checker le model dispo dans config pour dagster (on stop la pipeline) ---
    if model_name not in MODELS_CONFIG.keys():
        raise ValueError(f"le model {model_name} n'est pas configuré ")
    
    # Lire les données traitées depuis PRCESSED_DIR
    TARGET_FILE_PATH = Path(PROCESSED_DIR ,"bordeaux_conso_mwh.parquet")
    df = pd.read_parquet(TARGET_FILE_PATH)

    # --- copier le df afin d'eviter de corempre les données ----
    df_cp = df.copy()

    # --- Renomer pour fb (et les autres histoire d'eviter un if unitile) ----
    df_cp = df_cp[['date', 'daily_conso_mgw']].rename(columns={'date':'ds', 'daily_conso_mgw':'y'})

    # --- Train / Test split ---
    train_size = int(len(df_cp) * 0.8)
    train = df_cp.iloc[:train_size]
    test = df_cp.iloc[train_size:]

    #--- charger la classe dynamiquement ---
    class_path = MODELS_CONFIG[model_name].get("class")  # "prophet.Prophet"
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)

    # --- params ---
    params = MODELS_CONFIG[model_name].get("params", {})

    # --- instancier ---
    model = ModelClass(**params)

    # --- train_model ---- 
    model.fit(train)

    # --- make prediction ---
    future = model.make_future_dataframe(periods=len(test), freq="D")
    forecast = model.predict(future)

    # --- récupérer juste les prédictions pour le test set ---
    yhat_test = forecast['yhat'].values[-len(test):]
    y_true = test['y'].values

    # --- compute metrics ---
    mae = mean_absolute_error(y_true, yhat_test)
    rmse = np.sqrt(mean_squared_error(y_true, yhat_test))
    r2 = r2_score(y_true, yhat_test)
    wape = (abs(y_true - yhat_test).sum() / abs(y_true).sum()) * 100
    
    metrics = {"mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "wape": float(wape),
    "model": model_name
    }
    
    # ---- save the model ----
    model_path = Path(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)

    # ---- save metrics ----
    metrics_path = Path(METRICS_DIR, f"{model_name}_metrics.json")
    pd.Series(metrics).to_json(metrics_path)

    # ---- save forecast ---- (utile pour Streamlit)
    forecast_path = Path(FORECAST_DIR, f"{model_name}_forecast.parquet")
    forecast.to_parquet(forecast_path, index=False)

    # ---- return useful objects for dagster ----
    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "forecast_path": str(forecast_path),
        "metrics": metrics,
    }



