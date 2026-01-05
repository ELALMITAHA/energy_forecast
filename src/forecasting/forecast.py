from pathlib import Path
import joblib

import pandas as pd

from utils.logger import logger
from configs.paths_config import BASE_DIR


class ModelForecaster:
    """
    Generate forecasts using a trained model and prepared dataset.

    This class handles loading the model and data, preparing the data using a
    specified preparator, running forecasts, and saving the results for monitoring
    and reporting purposes.

    Parameters
    ----------
    model_file_name : str
        File name of the serialized model to use for forecasting.

    data_file_name : str
        File name of the input dataset.

    input_model_folder : str or Path
        Directory containing the trained model files.

    input_data_folder : str or Path
        Directory containing the input data files.

    output_forecast_folder : str or Path
        Directory where forecasts will be saved.

    model_preparator_cls : class
        Class used to prepare data for the model.

    model_preparator_args : dict
        Arguments to initialize the data preparator.

    model_interface_cls : class
        Class providing the model interface (target/prediction column names).

    model_name : str
        Name of the model (used for reporting and saving forecasts).

    Behavior
    --------
    - Loads dataset and trained model from specified folders.
    - Prepares data using a dedicated preparator class.
    - Generates forecasts using the loaded model.
    - Saves forecasts to the specified output folder.
    - Logs all steps and errors without breaking the pipeline.
    """

    # ***** Initialization *****
    def __init__(
        self,
        model_file_name,
        data_file_name,
        input_model_folder,
        input_data_folder,
        output_forecast_folder,
        model_preparator_cls,
        model_preparator_args,
        model_interface_cls,
        model_name,
    ):
        self.model_file_name = model_file_name
        self.data_file_name = data_file_name
        self.input_model_folder = input_model_folder
        self.input_data_folder = input_data_folder
        self.output_forecast_folder = output_forecast_folder
        self.model_preparator_cls = model_preparator_cls
        self.model_preparator_args = model_preparator_args
        self.model_interface_cls = model_interface_cls
        self.model_name = model_name

    # ***** Load Dataset *****
    def _load_data(self):
        """
        Load input dataset from parquet file.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.

        Raises
        ------
        FileNotFoundError
            If the data file is not found.
        """
        path_to_data = Path(self.input_data_folder, self.data_file_name)
        try:
            df = pd.read_parquet(path_to_data)
            return df
        except FileNotFoundError:
            logger.error(
                f"[FORECAST] {self.data_file_name} not found in {self.input_data_folder}"
            )
            raise

    # ***** Load Model *****
    def _load_model(self):
        """
        Load a trained forecasting model from file.

        Returns
        -------
        object
            Deserialized model instance.

        Raises
        ------
        FileNotFoundError
            If the model file is not found.
        """
        path_to_model = Path(
            self.input_model_folder, self.model_name, self.model_file_name
        )
        try:
            model = joblib.load(path_to_model)
            return model
        except FileNotFoundError:
            logger.error(
                f"[FORECAST] {self.model_file_name} not found in {self.input_model_folder}"
            )
            raise

    # ***** Prepare Data *****
    def _prepare_data(self):
        """
        Prepare dataset for forecasting using the specified preparator class.

        Returns
        -------
        pd.DataFrame
            Prepared dataset ready for forecasting.
        bool
            Flag indicating whether data preparation passed validation.
        dict
            Data preparation report containing validation details.
        """
        df = self._load_data()

        preparator_args = dict(self.model_preparator_args)
        preparator_args["df"] = df

        preparator = self.model_preparator_cls(**preparator_args)
        df_prepared, flag, report = preparator.prepare_data()

        return df_prepared, flag, report

    # ***** Generate Forecast *****
    def forecast(self):
        """
        Run the full forecasting pipeline.

        Steps
        -----
        1. Prepare the data using the preparator.
        2. Load the trained model.
        3. Generate forecasts using the model interface.
        4. Save forecasts to the specified output folder.

        Returns
        -------
        pd.DataFrame or bool
            Forecast DataFrame if successful; False if data validation failed.
        dict, optional
            Data preparation report if validation fails.
        """
        # Prepare the data
        df_prepared, flag, report = self._prepare_data()
        if not flag:
            logger.error(
                "[FORECAST] Data validation failed during preparation. See report for details."
            )
            return flag, report

        # Load the trained model
        model = self._load_model()

        # Instantiate model interface and inject loaded model
        model_interface = self.model_interface_cls()
        model_interface.model = model

        # Generate forecasts
        forecast = model_interface.predict(df_prepared)

        # Ensure output folder exists
        forecast_dir = Path(self.output_forecast_folder, self.model_name)
        forecast_dir.mkdir(parents=True, exist_ok=True)

        # Save forecasts
        forecast_path = forecast_dir / "forecasts.parquet"
        forecast.to_parquet(forecast_path)
        logger.info(
            f"[FORECAST] Forecast data saved to {forecast_path.relative_to(BASE_DIR)}"
        )

        return forecast
