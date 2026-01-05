from pathlib import Path
import pandas as pd

from utils.logger import logger
from configs.paths_config import BASE_DIR, RAW_DIR, PROCESSED_DIR


def process_weather(
    input_file_weather_archive,
    input_file_weather_forecast,
    output_file_weather_full_processed,
    input_folder,
    output_folder,
):
    """
    Process and merge weather archive and forecast data, then save the cleaned output.

    Parameters
    ----------
    input_file_weather_archive : str
        Filename of the weather archive Parquet file located in the input folder.

    input_file_weather_forecast : str
        Filename of the weather forecast Parquet file located in the input folder.

    output_file_weather_full_processed : str
        Target filename for the merged and processed Parquet file to be saved
        in the output folder. Existing files with the same name will be overwritten.

    input_folder : str or pathlib.Path
        Directory where the raw Parquet files are located.

    output_folder : str or pathlib.Path
        Directory where the processed Parquet file will be saved.
        The directory is automatically created if it does not exist.

    Behavior
    --------
    - Builds full paths to the archive and forecast Parquet files.
    - Loads the raw Parquet files into pandas DataFrames.
    - Creates copies of the DataFrames to avoid modifying the original files.
    - Concatenates archive and forecast data into a single DataFrame.
    - Removes duplicate rows in the 'time' column (e.g., the current day appears twice).
    - Converts the 'time' column to datetime and renames it to 'date'.
    - Renames temperature columns for consistency: max, min, and mean.
    - Ensures the output directory exists before saving.
    - Saves the processed DataFrame as a Parquet file to the output folder.
    - Logs detailed information about the process for traceability and monitoring.

    Raises
    ------
    FileNotFoundError
        If either the archive or forecast input file does not exist in the input folder.

    RuntimeError
        If an unexpected error occurs during file reading, processing, or saving.

    Notes
    -----
    - Designed for production pipelines: ensures consistent weather data formatting.
    - Idempotent: overwrites existing processed files with the same name.
    - Logs all relevant metadata for monitoring and debugging.
    - Can be integrated into larger data ingestion or ETL workflows.
    """
    # Build the full path to the source files in the 'raw' folder
    path_for_loading_archive = Path(input_folder, input_file_weather_archive)
    path_for_loading_forecast = Path(input_folder, input_file_weather_forecast)

    # Read the Parquet files with error handling
    try:
        # ********** Load the file into a dataframe *************
        df_archive = pd.read_parquet(path_for_loading_archive)
        df_forecast = pd.read_parquet(path_for_loading_forecast)

        # ********** Copy the dataframe to avoid modifying the original (protect the source file) ************
        df_archive_cp = df_archive.copy()
        df_forecast_cp = df_forecast.copy()

        # ********** Concatenate both dataframes *************
        df = pd.concat([df_archive_cp, df_forecast_cp])

        # ********* Today's day appears twice (archive + forecast), so remove duplicates **********
        df = df.drop_duplicates(subset="time")

        # ********* Convert the 'time' column **************
        df["time"] = pd.to_datetime(df["time"])

        # *********** Rename columns cleanly **************
        df = df.rename(
            columns={
                "time": "date",
                "temperature_2m_max": "temperature_max",
                "temperature_2m_min": "temperature_min",
                "temperature_2m_mean": "temperature_mean",
            }
        )

        # ************* Save to parquet *****************
        save_path = Path(output_folder, output_file_weather_full_processed)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)

        logger.info(
            f"[PROCESS WEATHER] {input_file_weather_archive} and "
            f"{input_file_weather_forecast} processed"
        )
        logger.info(
            f"[PROCESS WEATHER] Processed file saved to {save_path.relative_to(BASE_DIR)} "
        )

    except FileNotFoundError as e:
        # If one of the files does not exist, log and raise an explicit error
        logger.error("[PROCESS WEAHTHER] Weather data file not found in DATA/RAW")
        raise FileNotFoundError(
            "Weather data ingestion failed. Check the files in DATA/RAW."
        ) from e

    except Exception as e:
        # Capture any unexpected error during the reading process
        logger.error(
            f"[PROCESS WEAHTHER] Unexpected error while reading weather files: {e}"
        )
        raise RuntimeError(
            f"Unexpected error reading {path_for_loading_forecast} or {path_for_loading_archive}: {e}"
        ) from e


if __name__ == "__main__":
    process_weather(
        input_file_weather_forecast="bordeaux_weather_forecast.parquet",
        input_file_weather_archive="bordeaux_weather_archive.parquet",
        output_file_weather_full_processed="bordeaux_weather_full.parquet",
        output_folder=PROCESSED_DIR,
        input_folder=RAW_DIR,
    )
