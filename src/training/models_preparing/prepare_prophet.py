from utils.validate_data import DataValidator
import pandas as pd


class ProphetPreparator:
    """
    Prepares and validates time series data for Prophet model training.

    This class handles:
    - Renaming columns to Prophet conventions ("date" → "ds", target → "y").
    - Feature engineering (Heating/Cooling Degree Days: HDD/CDD).
    - Data validation using `DataValidator`.
    - Saving a data quality report for monitoring.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataframe containing consumption and weather features.

    target_col_name : str
        Name of the target column to predict (e.g., 'daily_conso_kwh').

    date_col_name : str
        Name of the date column.

    is_training_data : bool
        Flag indicating whether the data is for training or forecasting.

    Raises
    ------
    ValueError
        If data validation fails after preparation.

    Notes
    -----
    - Designed for production pipelines: idempotent and fault-tolerant.
    - Structured logging and validation report enable traceability.
    - Feature engineering ensures consistency of weather-derived regressors (HDD/CDD).

    Example
    -------
    >>> preparator = ProphetPreparator(
    >>>     df=my_raw_df,
    >>>     target_col_name="daily_conso_kwh",
    >>>     date_col_name="date",
    >>>     is_training_data=True
    >>> )
    >>> df_prepared, flag, report = preparator.prepare_data()
    """

    # ***** Initialization *****
    def __init__(self, df: pd.DataFrame, target_col_name: str, date_col_name: str, is_training_data: bool):
        self.df = df
        self.target_col_name = target_col_name
        self.date_col_name = date_col_name
        self.is_training_data = is_training_data

    # ***** Column renaming and initial cleaning *****
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns and drop missing target values if training.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to clean.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with columns renamed for Prophet.
        """
        if self.is_training_data:
            df = df.dropna(subset=[self.target_col_name])
        else:
            df = df.drop(columns=[self.target_col_name], errors="ignore")

        df = df.rename(columns={self.date_col_name: "ds", self.target_col_name: "y"})
        return df

    # ***** Feature engineering *****
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Heating Degree Days (HDD) and Cooling Degree Days (CDD) features.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with temperature column.

        Returns
        -------
        pd.DataFrame
            Dataframe with added HDD and CDD columns.
        """
        T_base_heat = 18
        T_base_cool = 22

        df["HDD"] = (T_base_heat - df["temperature_mean"]).clip(lower=0)
        df["CDD"] = (df["temperature_mean"] - T_base_cool).clip(lower=0)
        return df

    # ***** Full preparation pipeline *****
    def prepare_data(self):
        """
        Execute the full data preparation pipeline: cleaning, feature engineering,
        and validation.

        Returns
        -------
        pd.DataFrame
            Prepared dataframe ready for Prophet training or forecasting.

        bool
            Flag indicating if data validation passed.

        dict
            Data validation report.
        """
        # ********** Initial cleaning **********
        df = self._prepare_data(self.df)

        # ********** Feature engineering **********
        df = self._feature_engineering(df)

        # ********** Data validation **********
        validator = DataValidator(
            df=df,
            date_col_name="ds",
            target_col_name="y",
            is_training_data=self.is_training_data,
            regressors_list=["HDD", "CDD"],
        )
        flag, report = validator.validate_data()

        # ********** Return columns according to training/forecast **********
        if self.is_training_data:
            return df[["ds", "y", "HDD", "CDD"]], flag, report
        else:
            return df[["ds", "HDD", "CDD"]], flag, report

