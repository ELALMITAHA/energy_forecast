import requests
from pathlib import Path
import time

import pandas as pd

from configs.api_config import (
    OPEN_METEO_BASE_URL,
    WEATHER_SOURCES,
)
from utils.logger import logger
from configs.paths_config import BASE_DIR


def fetch_weather(
    api_name,
    output_file_name,
    output_folder,
    max_retries=3,
    retry_delay=5,
):
    """
    Fetch daily weather data from the Open-Meteo API and save it as a parquet file
    in a specified output directory.

    Parameters
    ----------
    api_name : str
        Key identifying the API configuration defined in `WEATHER_SOURCES` and
        `OPEN_METEO_BASE_URL`.
        Examples:
        - "bordeaux_weather_archive"
        - "bordeaux_weather_forecast"

    output_file_name : str
        Name of the output parquet file (including '.parquet').
        The file will be overwritten if it already exists.

    output_folder : str or pathlib.Path
        Directory where the parquet file will be saved.
        The directory is automatically created if it does not exist.

    max_retries : int, optional
        Maximum number of retry attempts if the HTTP request fails.
        Default is 3.

    retry_delay : int or float, optional
        Delay (in seconds) before retrying after a failed request.
        Default is 5.

    Behavior
    --------
    - Loads API configuration (latitude, longitude, variables, dates) from
      `WEATHER_SOURCES`.
    - Sends an HTTP GET request with timeout and retry logic for network/API errors.
    - Validates that the response contains a "daily" field and raises a ValueError
      if missing.
    - Converts the returned daily data into a pandas DataFrame.
    - Logs ingestion metadata including row count, date range, and HTTP status.
    - Ensures the output directory exists before saving the parquet file.
    - Saves the resulting data as a parquet file, overwriting any existing file.

    Raises
    ------
    ValueError
        If the "daily" key is missing or the response format is invalid.

    requests.RequestException
        If all retry attempts fail due to HTTP or network errors.

    Notes
    -----
    - Designed for production ingestion pipelines with robust error handling.
    - Idempotent: repeated executions produce the same output file.
    - Supports both historical archive and forecast endpoints.
    - Logging provides traceability and monitoring for MLOps workflows.
    """

    for attempt in range(1, max_retries + 1):
        try:
            # ********** Request **********
            source = WEATHER_SOURCES.get(api_name, {})
            url = OPEN_METEO_BASE_URL.get(api_name, {})
            params = {
                "latitude": source["lat"],
                "longitude": source["lon"],
                "daily": source["daily"],
                "start_date": source.get("start_date"),
                "end_date": source.get("end_date"),
                "timezone": "auto",
            }
            response = requests.get(url, params, timeout=15)
            response.raise_for_status()

            data = response.json()

            if "daily" not in data:
                raise ValueError("No 'daily' data in API response")

            df = pd.DataFrame(data["daily"])

            # ********* Logging **********
            if not df.empty:
                min_date = df["time"].min()
                max_date = df["time"].max()
                logger.info(
                    f"[INGESTION WEATHER] Loaded {len(df)} rows from API '{api_name}' "
                )
                logger.info(
                    f"[INGESTION WEATHER] Dates from {min_date} to {max_date} | HTTP {response.status_code}"
                )
            else:
                logger.warning(
                    f"[INGESTION WEATHER] No rows loaded from API '{api_name}' | HTTP {response.status_code}"
                )
                logger.warning("[INGESTION WEATHER] API returned empty data")

            # ********* Save file parquet ***********
            save_path = Path(output_folder, output_file_name)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path, index=False)
            logger.info(
                f"[INGESTION WEATHER] Weather data saved to {save_path.relative_to(BASE_DIR)}"
            )

            break

        except requests.RequestException as e:
            logger.error(f"[INGESTION WEATHER] Request failed: {e}")
            if attempt < max_retries:
                logger.info(f"[INGESTION WEATHER] Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise
        except ValueError as e:
            logger.error(f"[INGESTION WEATHER] Data error: {e}")
