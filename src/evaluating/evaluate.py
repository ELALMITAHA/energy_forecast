from pathlib import Path
import joblib

import pandas as pd

from utils.logger import logger
from configs.paths_config import BASE_DIR

from utils.compute_metrics import evaluate_rolling_window
from utils.save_files import save_metrics


class ModelEvaluator:
    """
    Evaluate a trained forecasting model on a prepared dataset and compute metrics.

    This class handles loading the model and data, preparing the data using a
    specified preparator, running predictions, computing evaluation metrics,
    and saving the results for monitoring and reporting purposes.

    Parameters
    ----------
    model_file_name : str
        File name of the serialized model to evaluate.

    data_file_name : str
        File name of the input dataset.

    input_model_folder : str or Path
        Directory containing the model files.

    input_data_folder : str or Path
        Directory containing the input data files.

    output_metrics_folder : str or Path
        Directory where evaluation metrics will be saved.

    model_preparator_cls : class
        Class used to prepare data for the model.

    model_preparator_args : dict
        Arguments to initialize the data preparator.

    model_interface_cls : class
        Class providing the model interface (target/prediction column names).

    model_name : str
        Name of the model (used for reporting and saving metrics).

    Behavior
    --------
    - Loads dataset and model from specified folders.
    - Prepares data using a dedicated preparator class.
    - Computes forecasts using the loaded model.
    - Evaluates predictions with rolling-window metrics.
    - Saves metrics report to the specified output folder.
    - Logs all steps and errors without breaking the evaluation pipeline.
    """

    # ***** Initialization *****
    def __init__(
        self,
        model_file_name,
        data_file_name,
        input_model_folder,
        input_data_folder,
        output_metrics_folder,
        model_preparator_cls,
        model_preparator_args,
        model_interface_cls,
        model_name,
    ):
        self.model_file_name = model_file_name
        self.data_file_name = data_file_name
        self.input_model_folder = input_model_folder
        self.input_data_folder = input_data_folder
        self.output_metrics_folder = output_metrics_folder
        self.model_preparator_cls = model_preparator_cls
        self.model_preparator_args = model_preparator_args
        self.model_interface_cls = model_interface_cls
        self.model_name = model_name

    # ***** Load Dataset *****
    def _load_data(self):
        """
        Load dataset from parquet file.

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
                f"[EVALUATION] {self.data_file_name} not found in {self.input_data_folder}"
            )
            raise

    # ***** Load Model *****
    def _load_model(self):
        """
        Load a trained model from file.

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
                f"[EVALUATION] {self.model_file_name} not found in {self.input_model_folder}"
            )
            raise

    # ***** Prepare Data *****
    def _prepare_data(self):
        """
        Prepare data for model evaluation using the specified preparator class.

        Returns
        -------
        pd.DataFrame
            Prepared dataset ready for prediction.
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

    # ***** Evaluate Model *****
    def evaluate(self):
        """
        Run the full evaluation pipeline.

        Steps
        -----
        1. Prepare the data using the preparator.
        2. Load the trained model.
        3. Compute predictions on the prepared dataset.
        4. Evaluate metrics using rolling-window computations.
        5. Save metrics report to the specified output folder.

        Returns
        -------
        bool
            Global evaluation flag; False if data validation failed.
        dict
            Evaluation report including metrics and any data preparation issues.
        """
        # Prepare the data
        df_prepared, flag, report = self._prepare_data()

        if not flag:
            logger.error(
                "[EVALUATION] Data validation failed during evaluation. See report for details."
            )
            return flag, report

        # Load trained model
        model = self._load_model()

        # Predict
        forecast = model.predict(df_prepared)

        # Instantiate interface for column names
        model_interface = self.model_interface_cls()
        ytrue = df_prepared[model_interface.target_col]
        yhat = forecast[model_interface.pred_col]

        # Compute evaluation metrics
        metrics = evaluate_rolling_window(
            y_true=ytrue, y_pred=yhat, window_size=60, seasonality=7
        )
        logger.info(f"[EVALUATION] Metrics computed: {metrics}")

        # Ensure output folder exists
        self.output_metrics_folder = Path(self.output_metrics_folder)
        self.output_metrics_folder.mkdir(parents=True, exist_ok=True)

        # Save metrics
        save_metrics(
            metrics=metrics,
            output_dir=self.output_metrics_folder,
            model_name=self.model_name,
            model_version="v1",
        )
        logger.info(
            f"[EVALUATION] Metrics report saved to {self.output_metrics_folder.relative_to(BASE_DIR)}"
        )

        # Update report
        report.update({"metrics": metrics})

        return True, report
