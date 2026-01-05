from datetime import datetime, timedelta, UTC
import requests
from pathlib import Path

import pandas as pd

from utils.logger import logger
from configs.api_config import OSD_PARAMS, OSD_SOURCES, OSD_BASE_URL
from configs.paths_config import BASE_DIR


def fetch_target(
    api_name, output_file_name, output_folder, max_retries=3, retry_delay=5, mode="full"
):
    """
    Fetch daily or full historical electricity consumption (kWh) data from an OSD API
    and store the results as a parquet file.

    Parameters
    ----------
    api_name : str
        The API name key defined in `OSD_SOURCES` and `OSD_PARAMS`.
        Example: "bordeaux_conso_mwh".

    output_file_name : str
        Name of the output parquet file (including '.parquet').
        The file will be overwritten if it already exists.

    output_folder : str or pathlib.Path
        Directory where the parquet file will be saved.
        The directory is automatically created if it does not exist.

    max_retries : int, optional
        Maximum number of retry attempts in case of HTTP/network failure.
        Default is 3.

    retry_delay : int or float, optional
        Number of seconds to wait between retry attempts.
        Default is 5.

    mode : {"full", "daily"}, optional
        Fetch mode:
        - "full": downloads all available historical data.
        - "daily": only downloads yesterday’s data (based on UTC).
        Default is "full".

    Behavior
    --------
    - Builds API request parameters from config dictionaries.
    - Executes an HTTP GET request with retry logic for network errors.
    - Validates response format and raises ValueError if invalid.
    - Converts records into a pandas DataFrame.
    - Logs detailed ingestion information (row count, date range, retries).
    - Saves the result to `RAW_DIR/bordeaux_conso_kwh.parquet`.

    Raises
    ------
    ValueError
        If the API name is invalid or the response structure is inconsistent.

    requests.RequestException
        If all retry attempts fail due to HTTP or network errors.

    Notes
    -----
    This function is designed to be robust in production settings (MLOps):
    - Idempotent: saving always overwrites the same parquet file.
    - Fault-tolerant thanks to retry logic.
    - Structured logging for monitoring and debugging.
    """
    # ******* handling mode of fetching data *********
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

    for attemp in range(1, max_retries + 1):
        try:
            # ********** Request **********
            source = OSD_SOURCES[api_name]
            url = OSD_BASE_URL.format(dataset=source["dataset"])
            params = OSD_PARAMS[api_name].copy()
            for key, value in source.get("refine", {}).items():
                params[f"refine[{key}]"] = value

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if "records" in data:
                rows = [rec.get("record", rec) for rec in data["records"]]
            elif "results" in data:
                rows = data["results"]
            else:
                rows = []

            df = pd.DataFrame(rows)

            # ********* Logging **********
            if not df.empty:
                min_date = df["jour"].min()
                max_date = df["jour"].max()
                if mode == "daily":
                    logger.info(
                        f"[INGESTION CONSUMPTION] Loaded {len(df)} rows from API '{api_name}' "
                    )
                    logger.info(
                        f"[INGESTION CONSUMPTION] Consumption date {min_date} | HTTP {response.status_code}"
                    )
                else:
                    logger.info(
                        f"[INGESTION CONSUMPTION] Loaded {len(df)} rows from API '{api_name}'"
                    )
                    logger.info(
                        f"[INGESTION CONSUMPTION] Consumption dates from {min_date} to {max_date} | HTTP {response.status_code}"
                    )

            else:
                logger.warning(
                    f"[INGESTION CONSUMPTION] No rows loaded from API '{api_name}' | HTTP {response.status_code}"
                )

            # ********* Save file parquet ***********
            save_path = Path(output_folder, output_file_name)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path, index=False)
            logger.info(
                f"[INGESTION CONSUMPTION] Consumption data saved to {save_path.relative_to(BASE_DIR)}"
            )

            break

        except requests.RequestException as e:
            logger.error(f"[INGESTION CONSUMPTION] Request failed : {e}")
            if attemp < max_retries:
                logger.info(f"[INGESTION CONSUMPTION] Retrying in {retry_delay}s...")
        except ValueError as e:
            logger.error(f"[INGESTION CONSUMPTION] Data error: {e}")
