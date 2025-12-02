from dagster import job, op, Definitions
from flows.ingestion.fetch_target import fetch_target_mwh
from flows.process.process_target import process_target
from utils.logger import logger

@op
def ingest_op():
    fetch_target_mwh("bordeaux_conso_mwh")
    return "done"  # simple signal pour la dépendance

@op
def process_op(ingest_result):  # <- recevoir la sortie de ingest_op
    report = process_target("bordeaux_conso_mwh.parquet")
    logger.info(f"Pipeline OK - report: {report}")

@job
def full_pipeline():
    process_op(ingest_op())  # process_op dépend de ingest_op

defs = Definitions(
    jobs=[full_pipeline]
)



