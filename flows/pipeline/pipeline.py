# pipelines/run_full_pipeline.py
from flows.ingestion.fetch_target import fetch_target_mwh
from flows.process.process_target import process_target
from utils.logger import logger

def main():
    logger.info("=== PIPELINE START ===")

    fetch_target_mwh("bordeaux_conso_mwh")  # écrit dans data/raw/

    report = process_target("bordeaux_conso_mwh.parquet")  # lit raw/, transforme, valide, sauvegarde

    logger.info(f"Pipeline OK - report: {report}")
    logger.info("=== PIPELINE END ===")

if __name__ == "__main__":
    main()