from dagster import op, job, Definitions

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

from src.training.update_retrain_flag import update_retrain_flag  # <-- typo corrigée
from src.training.load_retrain_flag import load_retrain_flag

# ----------------------
# OPS
# ----------------------

@op
def ingest_op() -> bool:
    logger.info("[DAGSTER] ==================================== Starting pipeline ======================================== ")
    logger.info("[DAGSTER] Starting ingestion step...")
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
    return True

@op
def process_op(start: bool) -> bool:
    logger.info("[DAGSTER] Starting processing step...")
    process_target(
        input_file_name="bordeaux_conso_kwh.parquet",
        output_file_name="bordeaux_conso_kwh.parquet",
        input_folder=RAW_DIR,
        output_folder=PROCESSED_DIR,
    )
    process_weather(
        input_file_weather_forecast="bordeaux_weather_forecast.parquet",
        input_file_weather_archive="bordeaux_weather_archive.parquet",
        output_file_weather_full_processed="bordeaux_weather_full.parquet",
        input_folder=RAW_DIR,
        output_folder=PROCESSED_DIR,
    )
    return True

@op
def merge_op(start: bool) -> bool:
    logger.info("[DAGSTER] Starting merging step...")
    merge_target_weather(
        input_target_file_name="bordeaux_conso_kwh.parquet",
        input_weather_file_name="bordeaux_weather_full.parquet",
        output_full_data_name="full_data.parquet",
        input_folder=PROCESSED_DIR,
        output_folder=PROCESSED_FINAL_DIR,
    )
    return True

@op
def decide_retrain_op(start: bool) -> bool:
    retrain_flag = load_retrain_flag(
        retrain_flag_file_name="retrain_flag.json",
        hf_repo_id="tahaelalmi/energy-forecast-artifacts",
    )
    if retrain_flag:
        logger.info("[DAGSTER] Retrain flag is True → model will be retrained.")
    else:
        logger.info("[DAGSTER] Retrain flag is False → skipping training.")
    return retrain_flag

@op
def train_op(start: bool, should_retrain: bool) -> str | None:
    if not should_retrain:
        logger.info("[DAGSTER] Train step skipped because retrain flag is False.")
        return None

    logger.info("[DAGSTER] Starting training step...")
    ModelTrainer(
        data_file_name="full_data.parquet",
        target_col_name="daily_conso_kwh",
        date_col_name="date",
        features_needed=["date", "daily_conso_kwh", "temperature_mean"],
        model_interface_cls=ProphetInterface,
        model_data_prepar_validate=ProphetPreparator,
        model_name="prophet",
    ).train_and_tune()

    best_model_path = "models/prophet/best_model.pkl"
    logger.info(f"[DAGSTER] Best model saved at {best_model_path}")

    return best_model_path

@op
def evaluate_op(model_path: str | None) -> dict:
    logger.info(f"[Dagster] Using model at {model_path}")

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

@op
def update_metrics_op(metrics: dict) -> bool:
    logger.info("[DAGSTER] Starting updating metrics on HF step...")
    updater = MetricHfUpdate(
            model_name="prophet",
            hf_metrics_dir="metrics",
            hf_repo_id="tahaelalmi/energy-forecast-artifacts",
            metrics_file_name="metrics.parquet",
        )
    updater.update_hf_db()

    logger.info("[METRICS UPDATE] Metrics successfully pushed to HuggingFace.")
    return True

@op
def update_retrain_flag_op(metrics: dict) -> bool:
    logger.info("[DAGSTER] Starting updating retraining flag step...")
    update_retrain_flag(
        output_flag_file="retrain_flag.json",
        output_flag_folder=RETRAIN_FLAG_DIR,
        metrics=metrics,
        metric_key="mase_window",
        hf_repo=None,
        hf_path_in_repo=None,
    )
    logger.info("[DAGSTER] Retrain flag updated based on latest metrics.")
    return True

@op
def forecast_op(metrics: dict):
    logger.info("[DAGSTER] Starting forecasting step...")
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


# ----------------------
# JOB
# ----------------------

@job
def full_pipeline():
    i = ingest_op()
    p = process_op(i)
    m = merge_op(p)

    should_retrain = decide_retrain_op(m)
    model_path = train_op(m, should_retrain)

    metrics = evaluate_op(model_path)
    update_retrain_flag_op(metrics)
    update_metrics_op(metrics) 
    forecast_op(metrics)
# ----------------------
# DEFINITIONS
# ----------------------

defs = Definitions(jobs=[full_pipeline])


