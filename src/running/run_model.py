import os 
from dotenv import load_dotenv
from pathlib import Path
import joblib

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


from utils.compute_metrics import evaluate_rolling_window
from utils.save_files import save_metrics
from utils.save_files import save_validation_report

from utils.logger import logger
from configs.paths_config import BASE_DIR,MODELS_DIR,DATA_QUALITY_DIR


class ModelRunner:
    """
        Class to run forecasts and evaluations using a trained model and prepared dataset.

        This class handles:
        - Loading input dataset and trained model (HF priority, local fallback)
        - Preparing data with a specified preparator class
        - Generating forecasts
        - Evaluating metrics with rolling-window computations
        - Saving forecasts and metrics for monitoring

        Parameters
        ----------
        model_file_name : str
            Serialized model file name.

        data_file_name : str
            Input dataset file name.

        hf_repo_id : str
            Hugging Face repository ID for model download.

        output_forecast_folder : str or Path
            Directory where forecast files are saved.

        output_metrics_folder : str or Path
            Directory where evaluation metrics are saved.

        input_data_folder : str or Path
            Directory containing input data files.

        model_preparator_cls : class
            Class used to prepare the dataset for forecasting.

        model_preparator_args : dict
            Arguments for initializing the data preparator.

        model_interface_cls : class
            Class providing model interface (column names and prediction methods).

        model_name : str
            Model name used for logging, reporting, and saving forecasts.

        Raises
        ------
        FileNotFoundError
            If model or data files are missing.

        ValueError
            If data preparation or validation fails.

        RuntimeError
            If unexpected errors occur during evaluation or forecasting.

        Notes
        -----
        - Idempotent: repeated calls produce the same outputs and overwrite existing parquet files.
        - Robust: fallback from HF to local models, structured logging, and fault-tolerant data preparation.
        - Designed for production (MLOps) pipelines for monitoring and traceability.

        Example
        -------
        >>> runner = ModelRunner(
        >>>     model_file_name="my_model.pkl",
        >>>     data_file_name="prepared_data.parquet",
        >>>     hf_repo_id="my-hf-repo",
        >>>     output_forecast_folder="outputs/forecasts",
        >>>     output_metrics_folder="outputs/metrics",
        >>>     input_data_folder="outputs/processed_data",
        >>>     model_preparator_cls=MyPreparator,
        >>>     model_preparator_args={},
        >>>     model_interface_cls=MyModelInterface,
        >>>     model_name="my_model"
        >>> )
        >>> metrics = runner.evaluate()
        >>> forecast_df = runner.forecast()
    """

    # ***** Initialization *****
    def __init__(
        self,
        model_file_name,
        data_file_name,
        hf_repo_id,
        output_forecast_folder,
        output_metrics_folder,
        input_data_folder,
        model_preparator_cls,
        model_preparator_args,
        model_interface_cls,
        model_name,
    ):
        self.model_file_name = model_file_name
        self.data_file_name = data_file_name
        self.input_data_folder = input_data_folder
        self.hf_repo_id = hf_repo_id
        self.output_forecast_folder = output_forecast_folder
        self.output_metrics_folder = output_metrics_folder
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
        Load a trained model with priority HF, fallback to local pipeline output.

        Returns
        -------
        object
            Deserialized model instance.

        Raises
        ------
        FileNotFoundError
            If no model is found on HF or locally.
        """
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")

        # Local fallback path (produit par le pipeline)
        local_model_path = MODELS_DIR / self.model_name / self.model_file_name

        # 1️⃣ Essayer HF en priorité
        try:
            hf_path_to_model = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"{self.model_name}/{self.model_file_name}",
                repo_type="model",
                token=hf_token,
                force_download=True,
            )
            logger.info(f"[MODEL RUNNER] [LOADING MODEL] Loaded model from HF: {hf_path_to_model}")
            return joblib.load(hf_path_to_model)

        except EntryNotFoundError:
            logger.warning(f"[MODEL RUNNER] [LOADING MODEL] No model founded in HF repo")


        except Exception as e:
            logger.warning(f"[MODEL RUNNER] [LOADING MODEL] HF model download failed: {e}")

        # 2️⃣ Fallback local
        if local_model_path.exists():
            logger.warning(f"[MODEL RUNNER] [LOADING MODEL] Using local model fallback: {local_model_path.relative_to(BASE_DIR)}")
            return joblib.load(local_model_path)

        # 3️⃣ Aucun modèle trouvé → log error et raise
        logger.error(
            f"[[MODEL RUNNER] [LOADING MODEL] No model available. Tried HF and local path: {local_model_path.relative_to(BASE_DIR)}"
        )
        raise FileNotFoundError(
            f"Cannot load model '{self.model_file_name}' from HF or local path."
        )

    # ***** Prepare Data *****
    def _prepare_data(self,is_training_data):
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
        df_raw = self._load_data()

        preparator_args = dict(self.model_preparator_args)
        preparator_args["df"] = df_raw
        preparator_args["is_training_data"] = is_training_data

        preparator = self.model_preparator_cls(**preparator_args)
        df_prepared, flag, report = preparator.prepare_data()

        return df_prepared, df_raw, flag, report
    
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
        df_prepared,_ , flag, report = self._prepare_data(is_training_data=True)

        if not flag:
            logger.error(
                "[MODEL RUNNER] [EVALUATION] Data validation failed during evaluation. See report for details."
            )
            return flag, report
        
        # Save data validation report
        save_validation_report(
            report=report,
            output_path=DATA_QUALITY_DIR,
            filename="data_quality_report.json",
        )

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
        logger.info(f"[MODEL RUNNER] [EVALUATION] Metrics computed: {metrics}")

        # Save metrics
        save_metrics(
            metrics=metrics,
            output_dir=self.output_metrics_folder,
            model_name=self.model_name,
        )
        logger.info(
            f"[EVALUATION] Metrics report saved to {self.output_metrics_folder.relative_to(BASE_DIR)}"
        )

        return metrics

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
        df_prepared,df_raw ,flag, report = self._prepare_data(is_training_data=False)
        if not flag:
            logger.error(
                "[MODEL RUNNER] [FORECAST] Data validation failed during preparation. See report for details."
            )
            return flag, report

        # Load the trained model
        model = self._load_model()

        # Instantiate model interface and inject loaded model
        model_interface = self.model_interface_cls()
        model_interface.model = model

        # Generate forecasts
        forecast = model_interface.predict(df_prepared)

        # add true values to prophet forecast output for y vs yhat plot 
        final_forcaste_df = pd.merge(
            left=forecast[["ds","yhat","yhat_lower","yhat_upper"]],
            right=df_raw[["date","daily_conso_kwh"]],
            left_on="ds",
            right_on="date",
            how="inner"
        )

         # Ensure output folder exists
        forecast_dir = Path(self.output_forecast_folder, self.model_name)
        forecast_dir.mkdir(parents=True, exist_ok=True)

        # Save forecasts
        forecast_path = forecast_dir / "forecasts.parquet"
        final_forcaste_df.to_parquet(forecast_path)
        logger.info(
            f"[MODEL RUNNER] [FORECAST] Forecast data saved to {forecast_path.relative_to(BASE_DIR)}"
        )


        return forecast
