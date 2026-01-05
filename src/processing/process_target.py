from pathlib import Path
import pandas as pd

from utils.logger import logger
from configs.paths_config import BASE_DIR, RAW_DIR, PROCESSED_DIR


def process_target(
    input_file_name,
    output_file_name,
    input_folder,
    output_folder,
):
    """
    Process and clean electricity consumption data from a raw Parquet file
    and save the processed output as a standardized Parquet file.

    Parameters
    ----------
    input_file_name : str
        Name of the raw consumption Parquet file to process
        (e.g., 'bordeaux_conso_kwh.parquet').

    output_file_name : str
        Name of the processed Parquet file to save
        (e.g., 'bordeaux_conso_kwh.parquet').
        Existing files will be overwritten.

    input_folder : str or pathlib.Path
        Directory where the raw Parquet file is located.

    output_folder : str or pathlib.Path
        Directory where the processed Parquet file will be saved.
        The directory is automatically created if it does not exist.

    Behavior
    --------
    - Builds full path to the raw input file.
    - Loads the Parquet file into a pandas DataFrame.
    - Creates a copy of the DataFrame to avoid modifying the original.
    - Drops debug or temporary columns if present.
    - Renames main columns to standardized names.
    - Converts the 'date' column to datetime (UTC-naive) and normalizes it.
    - Sorts the DataFrame by date and resets the index.
    - Ensures the output directory exists before saving.
    - Saves the cleaned DataFrame as a Parquet file to the output folder.
    - Logs detailed information for traceability and monitoring.

    Raises
    ------
    FileNotFoundError
        If the source file does not exist in the input folder.

    RuntimeError
        If an unexpected error occurs during file reading, processing, or saving.

    Notes
    -----
    - Designed for production pipelines: ensures consistent data formatting.
    - Idempotent: overwrites existing processed files with the same name.
    - Logs all relevant metadata for monitoring and debugging.
    - Can be integrated into larger data ingestion or ETL workflows.
    """

    # Build full path to the source file in the RAW folder
    path_for_loading = Path(input_folder, input_file_name)

    try:
        # ********** Load the file into a dataframe *************
        df = pd.read_parquet(path_for_loading)

        # ********** Copy the dataframe to avoid modifying the original *************
        df_cp = df.copy()

        # ********** Drop debug column if exists *************
        df_cp = df_cp.drop("range(date_heure,1day)", axis=1, errors="ignore")

        # ********** Rename main columns *************
        df_cp = df_cp.rename(columns={"jour": "date", "conso_jour": "daily_conso_kwh"})

        # ********** Convert 'date' column to datetime and normalize *************
        df_cp["date"] = pd.to_datetime(df_cp["date"]).dt.tz_localize(None)
        df_cp["date"] = pd.to_datetime(df_cp["date"]).dt.normalize()

        # ********** Sort by date and reset index *************
        df_cp = df_cp.sort_values("date").reset_index(drop=True)

        # ********** Save processed Parquet file *************
        save_path = Path(output_folder, output_file_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_cp.to_parquet(save_path, index=False)

        # ********** Log success *************
        logger.info(
            f"[PROCESS CONSUMPTION] {input_file_name} processed and saved to {save_path.relative_to(BASE_DIR)} "
        )

    except FileNotFoundError as e:
        # Log and raise explicit error if file does not exist
        logger.error(f"[PROCESS CONSUMPTION] File not found: {path_for_loading}")
        raise FileNotFoundError(
            f"Consumption data ingestion failed. Check the files in DATA/RAW."
        ) from e

    except Exception as e:
        # Log and raise unexpected errors
        logger.error(f"[PROCESS CONSUMPTION] Unexpected error while reading: {e}")
        raise RuntimeError(f"Unexpected error reading {path_for_loading}: {e}") from e


if __name__ == "__main__":
    process_target(
        input_file_name="bordeaux_conso_kwh.parquet",
        output_file_name="bordeaux_conso_kwh.parquet",
        input_folder=RAW_DIR,
        output_folder=PROCESSED_DIR,
    )
