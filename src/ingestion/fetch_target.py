from datetime import datetime, timedelta,UTC
import requests

import pandas as pd

from utils.logger import logger
from configs.api_config import OSD_PARAMS,OSD_SOURCES, BASE_URL
from configs.paths import RAW_DIR

def fetch_target_mwh(name_api, mode="full"):
    if name_api not in OSD_SOURCES:
        raise ValueError(f"API '{name_api}' non définie dans OSD_SOURCES")

    source = OSD_SOURCES[name_api]
    params = OSD_PARAMS[name_api].copy()

    # ========= WHERE clause selon mode ===========
    if mode == "daily":
        today_utc = datetime.utcnow().date()
        yesterday_utc = today_utc - timedelta(days=1)
        params["where"] = (
            f"date_heure >= '{yesterday_utc}T00:00:00' AND "
            f"date_heure < '{today_utc}T00:00:00'"
        )
    elif mode == "full":
        pass 
    else:
        raise ValueError("mode must be 'full' or 'daily'")

    # ========= Build URL et refinements ===========
    url = BASE_URL.format(dataset=source["dataset"])
    for key, value in source.get("refine", {}).items():
        params[f"refine[{key}]"] = value

    # ========= Request ============
    response = requests.get(url, params=params)

    if response.status_code != 200:
        logger.error(f"[INGESTION] HTTP {response.status_code} - {response.text}")
        response.raise_for_status()

    data = response.json()
    if "records" in data:
        rows = [rec.get("record", rec) for rec in data["records"]]
    elif "results" in data:
        rows = data["results"]
    else:
        rows = []

    df = pd.DataFrame(rows)

    # ======== Logging ========== 
    if not df.empty:
        min_date = df["jour"].min()
        max_date = df["jour"].max()
        if mode == "daily":
            logger.info(
                f"[INGESTION] Loaded {len(df)} rows from API '{name_api}' "
                f"for consumption date {min_date} | HTTP {response.status_code}"
            )
        else:
            logger.info(
                f"[INGESTION] Loaded {len(df)} rows from API '{name_api}' "
                f"for consumption dates from {min_date} to {max_date} | HTTP {response.status_code}"
            )
    else:
        logger.warning(f"[INGESTION] No rows loaded from API '{name_api}' | HTTP {response.status_code}")

    # ========== Sauvegarde parquet =================
    save_path = RAW_DIR / "bordeaux_conso_mwh.parquet"
    df.to_parquet(save_path, index=False)
    logger.info(f"[INGESTION] Saved API data to '{save_path}'")


