from pathlib import Path

import pandas as pd

from utils.logger import logger
from configs.paths_config import BASE_DIR


def merge_target_weather(
    input_target_file_name,
    input_weather_file_name,
    output_full_data_name,
    input_folder,
    output_folder,
):
    """
    Merge processed electricity consumption (target) data with processed weather data.

    Parameters
    ----------
    input_target_file_name : str
        Filename of the processed consumption (target) Parquet file located in the input folder.

    input_weather_file_name : str
        Filename of the processed weather Parquet file located in the input folder.

    output_full_data_name : str
        Filename for the merged output Parquet file.
        Existing files will be overwritten.

    input_folder : str or pathlib.Path
        Directory where the processed input files are located.

    output_folder : str or pathlib.Path
        Directory where the merged Parquet file will be saved.
        The directory is automatically created if it does not exist.

    Behavior
    --------
    - Builds full paths to the processed target and weather files.
    - Loads both Parquet files into pandas DataFrames.
    - Verifies that both DataFrames contain a 'date' column; raises a ValueError if missing.
    - Performs a left join on the 'date' column, keeping all weather dates.
      (Forecast weather dates may not have corresponding target values.)
    - Ensures the output directory exists before saving.
    - Saves the merged DataFrame as a Parquet file to the output folder.
    - Logs detailed information about the merge for traceability and monitoring.

    Raises
    ------
    FileNotFoundError
        If one of the input files does not exist.

    ValueError
        If the 'date' column is missing in one or both input datasets.

    RuntimeError
        If an unexpected error occurs during the merge or save process.

    Notes
    -----
    - The left join preserves future weather dates (forecast horizon) even if target values are missing.
    - Missing target values can later be used to distinguish training data from forecast data
      in downstream pipelines.
    - Designed for production pipelines with robust error handling and logging.
    - Idempotent: overwrites existing merged files with the same name.
    """
    try:
        # ****** Build paths to processed files **********
        path_for_loading_target = Path(input_folder, input_target_file_name)
        path_for_loading_weather = Path(input_folder, input_weather_file_name)

        # *********** Load data *************
        df_target = pd.read_parquet(path_for_loading_target)
        df_weather = pd.read_parquet(path_for_loading_weather)

        # *********** Check date column exists *************
        missing_columns = []
        if "date" not in df_target.columns:
            missing_columns.append(f"{input_target_file_name}")
        if "date" not in df_weather.columns:
            missing_columns.append(f"{input_weather_file_name}")

        if missing_columns:
            msg = f"'date' column missing in file(s): {', '.join(missing_columns)}"
            logger.error(f"[MERGE] {msg}")
            raise ValueError(msg)

        # *********** Merge *************
        df = pd.merge(df_weather, df_target, on="date", how="left")

        # ********** Save to ***********
        save_path = Path(output_folder, output_full_data_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        # ********** Logging ************
        logger.info(
            f"[MERGE] Successfully merged target '{input_target_file_name}' "
            f"with weather '{input_weather_file_name}' "
        )
        logger.info(f"[MERGE] Merged file saved to '{save_path.relative_to(BASE_DIR)}'")

    except FileNotFoundError as e:
        # Log and raise if any file is missing
        logger.error(f"[MERGE] File not found: {e.filename}")
        raise

    except ValueError as e:
        # Log and raise explicit errors (like missing date column)
        logger.error(f"[MERGE] {e}")
        raise

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"[MERGE] Unexpected error: {e}")
        raise RuntimeError(
            f"Unexpected error merging {input_target_file_name} and {input_weather_file_name}: {e}"
        ) from e
