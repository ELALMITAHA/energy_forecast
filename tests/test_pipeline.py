from configs.paths_config import (
    RAW_DIR,
    PROCESSED_DIR,
    PROCESSED_FINAL_DIR,
    METRICS_DIR,
    FORECAST_DIR,
    RETRAIN_FLAG_DIR
)
from utils.logger import logger

from src.ingesting.fetch_target import fetch_target
from src.ingesting.fetch_weather import fetch_weather

from src.processing.process_target import process_target
from src.processing.process_weather import process_weather
from src.processing.merge_target_weather import merge_target_weather

from src.training.models_interfaces.prophet_interface import ProphetInterface
from src.training.models_preparing.prepare_prophet import ProphetPreparator

from src.training.model_trainer import ModelTrainer
from src.running.run_model import ModelRunner
from src.updating_artifacts.metrics.updating_metrics import MetricHfUpdate

from src.training.update_retrain_flag import update_retrain_flag
from src.training.load_retrain_flag import load_retrain_flag

# ***** Fetch raw data from APIs *****
def fetch():
    """
    Fetch target consumption and weather data from APIs
    and save them as parquet files in RAW_DIR.
    """
    fetch_target(
        api_name="bordeaux_conso_kwh",
        output_file_name="bordeaux_conso_kwh.parquet",
        output_folder=RAW_DIR,
    )

    fetch_weather(
        api_name="bordeaux_weather_archive",
        output_file_name="bordeaux_weather_archive.parquet",
        output_folder=RAW_DIR,
    )

    fetch_weather(
        api_name="bordeaux_weather_forecast",
        output_file_name="bordeaux_weather_forecast.parquet",
        output_folder=RAW_DIR,
    )


# ***** Process raw data *****
def process():
    """
    Process raw target and weather data to clean and transform
    them into production-ready parquet files.
    """
    # Process target consumption
    process_target(
        input_file_name="bordeaux_conso_kwh.parquet",
        output_file_name="bordeaux_conso_kwh.parquet",
        input_folder=RAW_DIR,
        output_folder=PROCESSED_DIR,
    )

    # Process weather data (archive + forecast)
    process_weather(
        input_file_weather_forecast="bordeaux_weather_forecast.parquet",
        input_file_weather_archive="bordeaux_weather_archive.parquet",
        output_file_weather_full_processed="bordeaux_weather_full.parquet",
        input_folder=RAW_DIR,
        output_folder=PROCESSED_DIR,
    )


# ***** Merge target and weather data *****
def merge():
    """
    Merge processed target consumption with weather data
    and save as full_data.parquet in PROCESSED_FINAL_DIR.
    """
    merge_target_weather(
        input_target_file_name="bordeaux_conso_kwh.parquet",
        input_weather_file_name="bordeaux_weather_full.parquet",
        output_full_data_name="full_data.parquet",
        input_folder=PROCESSED_DIR,
        output_folder=PROCESSED_FINAL_DIR,
    )

# ***** Load retrain flag from HF *****
def load_flag():
    """
    Load retrain flag from Hugging Face repository.

    Returns
    -------
    bool
        True if retraining is required, False otherwise.
    """
    retrain_flag = load_retrain_flag(
        retrain_flag_file_name="retrain_flag.json",
        hf_repo_id="tahaelalmi/energy-forecast-artifacts",
    )
    
    return retrain_flag

# ***** Train model if retrain_flag is True *****
def train():
    """
    Train the Prophet model using ModelTrainer.
    Includes data preparation, hyperparameter tuning,
    and model serialization.
    """
    model_trainer = ModelTrainer(
        data_file_name="full_data.parquet",
        target_col_name="daily_conso_kwh",
        date_col_name="date",
        features_needed=["date", "daily_conso_kwh", "temperature_mean"],
        model_interface_cls=ProphetInterface,
        model_data_prepar_validate=ProphetPreparator,
        model_name="prophet",
    )

    model_trainer.train_and_tune()


# ***** Evaluate trained model *****
def evaluate():
    """
    Evaluate the trained model using ModelRunner.
    Computes metrics and returns them for further usage.
    
    Returns
    -------
    dict
        Evaluation metrics (e.g., rolling-window MASE).
    """
    metrics = ModelRunner(
        model_file_name="best_model.pkl",
        data_file_name="full_data.parquet",
        hf_repo_id="tahaelalmi/energy-forecast-models",
        input_data_folder=PROCESSED_FINAL_DIR,
        output_forecast_folder=FORECAST_DIR,
        output_metrics_folder=METRICS_DIR,
        model_preparator_cls=ProphetPreparator,
        model_interface_cls=ProphetInterface,
        model_preparator_args={
            "date_col_name": "date",
            "target_col_name": "daily_conso_kwh",
        },
        model_name="prophet",

    ).evaluate()

    return metrics

# ***** Update metrics on Hugging Face *****
def updatehfmetrics():
    """
    Upload evaluation metrics to Hugging Face dataset repository.
    """
    updater = MetricHfUpdate(
            model_name="prophet",
            hf_metrics_dir="metrics",
            hf_repo_id="tahaelalmi/energy-forecast-artifacts",
            metrics_file_name="metrics.parquet",
        )
    updater.update_hf_db()


# ***** Generate forecasts *****
def forecast():
    """
    Generate forecasts using the trained Prophet model
    and save them to FORECAST_DIR.
    """
    ModelRunner(
        model_file_name="best_model.pkl",
        data_file_name="full_data.parquet",
        hf_repo_id="tahaelalmi/energy-forecast-models",
        input_data_folder=PROCESSED_FINAL_DIR,
        output_forecast_folder=FORECAST_DIR,
        output_metrics_folder=METRICS_DIR,
        model_preparator_cls=ProphetPreparator,
        model_interface_cls=ProphetInterface,
        model_preparator_args={
            "date_col_name": "date",
            "target_col_name": "daily_conso_kwh",
        },
        model_name="prophet",

    ).forecast()


# ***** Update retrain flag based on metrics *****
def update_flag(metrics):
    """
    Update retrain_flag.json based on evaluation metrics
    and save to RETRAIN_FLAG_DIR.
    """
    update_retrain_flag(
        output_flag_file="retrain_flag.json",
        output_flag_folder=RETRAIN_FLAG_DIR,
        metrics=metrics,
        metric_key="mase_window",
        hf_repo=None,
        hf_path_in_repo=None,
    ) 


# ***** Full pipeline orchestration *****
def pipeline():
    """
    Orchestrate the full pipeline:
    1. Fetch raw data from APIs
    2. Process data
    3. Merge datasets
    4. Load retrain flag
    5. Train model (if required)
    6. Evaluate model
    7. Update metrics on HF
    8. Update retrain flag
    9. Generate forecasts
    """
    print("--- Start of pipeline ---")

    # ***** Fetch raw data *****
    fetch()

    # ***** Process raw data *****
    process()

    # ***** Merge target and weather *****
    merge()

    # ***** Load retrain flag *****
    retrain_flag = load_flag()

    # ***** Train model if flag is True *****
    if retrain_flag:
        train()
    else:
        logger.info(f"[DAGSTER] Train step skipped ... ")

    # ***** Evaluate model *****
    metrics = evaluate()

    # ***** Update HF metrics *****
    updatehfmetrics()
    
    # ***** Update retrain flag *****
    update_flag(metrics=metrics)

    # ***** Generate forecasts *****
    forecast()

    print("--- End of pipeline ---")


# ***** Execute pipeline if script is run directly *****
if __name__ == "__main__":
    pipeline()
