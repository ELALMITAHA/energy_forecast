from dagster import job, op, Definitions

from src.ingesting.fetch_target import fetch_target
from src.ingesting.fetch_weather import fetch_weather

from src.processing.process_target import process_target
from src.processing.process_weather import process_weather
from src.processing.merge_target_weather import merge_target_weather

from src.training.model_trainer import ModelTrainer

from src.evaluating.evaluate import ModelEvaluator
from src.forecasting.forecast import ModelForecaster
from src.forecasting.forecast import ModelForecaster

from src.training.models_interfaces.prophet_interface import ProphetInterface
from src.training.models_preparing.prepare_prophet import ProphetPreparator

from utils.logger import logger
from configs.paths_config import (
    RAW_DIR,
    PROCESSED_DIR,
    PROCESSED_FINAL_DIR,
    MODELS_DIR,
    METRICS_DIR,
    FORECAST_DIR,
)

# =====================
# DAGSTER IMPORTS
# =====================
from dagster import op, job, Definitions

# =====================
# OPS
# =====================


@op
def ingest_op() -> bool:
    logger.info(
        "[DAGSTER] ============================= Starting pipeline ====================================="
    )
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
def train_op(start: bool) -> bool:
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
    return True


@op
def evaluate_op(start: bool) -> bool:
    logger.info("[DAGSTER] Starting evaluation step...")
    ModelEvaluator(
        model_file_name="best_model.pkl",
        data_file_name="full_data.parquet",
        input_model_folder=MODELS_DIR,
        input_data_folder=PROCESSED_FINAL_DIR,
        output_metrics_folder=METRICS_DIR,
        model_preparator_cls=ProphetPreparator,
        model_preparator_args={
            "date_col_name": "date",
            "target_col_name": "daily_conso_kwh",
            "is_training_data": True,
        },
        model_name="prophet",
        model_interface_cls=ProphetInterface,
    ).evaluate()
    return True


@op
def forecast_op(start: bool):
    logger.info("[DAGSTER] Starting forecasting step...")
    ModelForecaster(
        model_file_name="best_model.pkl",
        data_file_name="full_data.parquet",
        input_model_folder=MODELS_DIR,
        input_data_folder=PROCESSED_FINAL_DIR,
        output_forecast_folder=FORECAST_DIR,
        model_preparator_cls=ProphetPreparator,
        model_preparator_args={
            "date_col_name": "date",
            "target_col_name": "daily_conso_kwh",
            "is_training_data": False,
        },
        model_interface_cls=ProphetInterface,
        model_name="prophet",
    ).forecast()
    logger.info(
        "[DAGSTER] ============================ Pipeline completed successfully. ================================="
    )
    return True


# =====================
# JOB
# =====================


@job
def full_pipeline():
    i = ingest_op()
    p = process_op(i)
    m = merge_op(p)
    t = train_op(m)
    e = evaluate_op(t)
    forecast_op(e)


# =====================
# DEFINITIONS
# =====================

defs = Definitions(jobs=[full_pipeline])
