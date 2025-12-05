from dagster import job, op, Definitions
from src.ingestion.fetch_target import fetch_target_mwh
from src.process.process_target import process_target
from src.training.models_trainer import train_and_forecast
from utils.logger import logger

@op
def ingest_op():
    fetch_target_mwh("bordeaux_conso_mwh")
    return "done"  # simple signal pour la dépendance

@op
def process_op(ingest_result):  # <- recevoir la sortie de ingest_op
    report = process_target("bordeaux_conso_mwh.parquet")
    logger.info(f"Pipeline OK - report: {report}")
    return "done"

@op
def train_op(process_result):
    # process_result ici n'est qu'un signal, les données sont déjà dans le parquet
    report = train_and_forecast("prophet")
    logger.info(f"Training OK - metrics: {report}")
    return report

@job
def full_pipeline():
    process_op(ingest_op())  # process_op dépend de ingest_op
    train_op(process_op(ingest_op()))
    
defs = Definitions(
    jobs=[full_pipeline]
)



