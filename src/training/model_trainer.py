from pathlib import Path
import pickle

import pandas as pd

from configs.paths_config import BASE_DIR, PROCESSED_DIR, MODELS_DIR, DATA_QUALITY_DIR
from utils.logger import logger

from src.training.tuners.optuna_tuner import OptunaTuner
from utils.save_files import save_validation_report
from utils.compute_metrics import mase_metric


class ModelTrainer:
    """
    Train and tune a forecasting model using prepared time series data.

    This class handles loading data, validating and preparing it, splitting into
    train/test sets, tuning the model with Optuna, and saving the best model
    for downstream evaluation or forecasting.

    Parameters
    ----------
    data_file_name : str
        Name of the input dataset parquet file.

    features_needed : list of str
        List of features required for training.

    target_col_name : str
        Name of the target column.

    date_col_name : str
        Name of the date column.

    model_interface_cls : class
        Class providing the model interface for training and prediction.

    model_data_prepar_validate : callable
        Class/function used to prepare and validate the raw data.

    model_name : str
        Name of the model (used for saving outputs and reporting).

    Behavior
    --------
    - Loads data from processed directory.
    - Validates and prepares the dataset.
    - Splits the dataset into train/test sets.
    - Tunes the model using Optuna for best hyperparameters.
    - Saves the best model to the models directory.
    - Logs all steps and errors without breaking the pipeline.
    """

    # ***** Initialization *****
    def __init__(
        self,
        data_file_name,
        features_needed,
        target_col_name,
        date_col_name,
        model_interface_cls,
        model_data_prepar_validate,
        model_name,
    ):
        self.data_file_name = data_file_name
        self.features_needed = features_needed
        self.target_col_name = target_col_name
        self.date_col_name = date_col_name
        self.model_data_prepar_validate = model_data_prepar_validate
        self.model_interface_cls = model_interface_cls
        self.model_name = model_name

    # ***** Load Dataset *****
    def _get_data(self):
        """
        Load parquet dataset from processed directory.

        Returns
        -------
        pd.DataFrame
            Loaded dataset containing the required features.

        Raises
        ------
        FileNotFoundError
            If the dataset file is missing.

        KeyError
            If required features are missing in the dataset.
        """
        path_to_data = Path(PROCESSED_DIR, "final", self.data_file_name)
        try:
            df = pd.read_parquet(path_to_data, columns=self.features_needed)
            return df
        except FileNotFoundError:
            logger.error(
                f"[Trainer] File {self.data_file_name} not found in {path_to_data}"
            )
            raise
        except ValueError as e:
            logger.error("[Trainer] Some required features are missing in parquet file")
            raise KeyError(f"Missing required features in {path_to_data}") from e
        except Exception:
            logger.error(f"[Trainer] Failed to load parquet: {path_to_data}")
            raise

    # ***** Train/Test Split *****
    def _train_test_split(self, df, train_size_ratio=0.8):
        """
        Split dataset into training and testing sets.

        Parameters
        ----------
        df : pd.DataFrame
            Prepared dataset.
        train_size_ratio : float, optional
            Proportion of data used for training (default=0.8).

        Returns
        -------
        pd.DataFrame
            Training dataset.
        pd.DataFrame
            Testing dataset.
        """
        train_size = int(len(df) * train_size_ratio)
        self.train_df = df.iloc[:train_size]
        self.test_df = df.iloc[train_size:]
        return self.train_df, self.test_df

    # ***** Train and Tune Model *****
    def train_and_tune(self):
        """
        Run the full training and tuning pipeline.

        Steps
        -----
        1. Load raw data from processed directory.
        2. Validate and prepare the dataset.
        3. Save data quality report.
        4. Split data into train/test sets.
        5. Tune the model using Optuna to find the best hyperparameters.
        6. Save the best trained model to the models directory.

        Returns
        -------
        None
            Saves the model to disk; logs all steps.
        """
        # Load raw data
        df_raw = self._get_data()

        # Prepare and validate data
        df_ready, flag, report = self.model_data_prepar_validate(
            df=df_raw,
            target_col_name=self.target_col_name,
            date_col_name=self.date_col_name,
            is_training_data=True,
        ).prepare_data()

        # Save data validation report
        save_validation_report(
            report=report,
            output_path=DATA_QUALITY_DIR,
            filename="data_quality_report.json",
        )

        if flag:
            logger.info("[Trainer] Data ready for training")
        else:
            raise ValueError("Data validation failed")

        # Split into training and testing sets
        train, test = self._train_test_split(df_ready)

        # Tune model with Optuna
        optuna_tuner = OptunaTuner(
            model_interface_cls=self.model_interface_cls,
            train_df=train,
            test_df=test,
            metric_fn=mase_metric,
            n_trials=50,
            direction="minimize",
        )
        best_model, _, _ = optuna_tuner.run()

        # Save the best model
        model_dir = Path(MODELS_DIR, self.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "best_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        logger.info(f"[Trainer] Best model saved at {model_path.relative_to(BASE_DIR)}")
