from utils.validate_data import DataValidator


class ProphetPreparator:
    """
    Prepares and validates time series data for Prophet model training.

    This class handles:
    - Column renaming to Prophet conventions.
    - Feature engineering (HDD/CDD calculations).
    - Data validation with `DataValidator`.
    - Saving a data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataframe containing consumption and weather features.

    target : str
        Name of the target column to predict (e.g., 'daily_conso_kwh').

    regressors : list of str
        List of additional regressor columns for training.

    Behavior
    --------
    - Renames columns for Prophet ("date" → "ds", target → "y").
    - Drops rows with missing target values.
    - Adds Heating Degree Days (HDD) and Cooling Degree Days (CDD).
    - Validates dataset structure, missing values, duplicates, and business rules.
    - Saves a data quality report to disk.
    - Raises an error if validation fails.

    Methods
    -------
    prepare_data_for_prophet()
        Execute full preparation pipeline and return the cleaned dataframe.
    """

    # ***** Initialization *****
    def __init__(
        self,
        df,
        target_col_name,
        date_col_name,
        is_training_data,
    ):
        self.df = df
        self.target_col_name = target_col_name
        self.date_col_name = date_col_name
        self.is_training_data = is_training_data

    # ***** Column renaming and initial cleaning *****
    def _prepare_data(self, df):

        if self.is_training_data:
            df = df.dropna(subset=self.target_col_name)
        else:
            df = df.drop(self.target_col_name, axis=1)

        df = df.rename(columns={self.date_col_name: "ds", self.target_col_name: "y"})

        return df

    # ***** Feature engineering *****
    def _feature_engineering(self, df):
        T_base_heat = 18
        T_base_cool = 22

        df["HDD"] = (T_base_heat - df["temperature_mean"]).clip(lower=0)
        df["CDD"] = (df["temperature_mean"] - T_base_cool).clip(lower=0)

        return df

    # ***** Full preparation pipeline *****
    def prepare_data(self):
        """
        Execute the full data preparation pipeline for Prophet.

        Returns
        -------
        pd.DataFrame
            Prepared and validated dataframe ready for Prophet training.

        Raises
        ------
        ValueError
            If the data validation step fails.
        """
        # ********** Initial cleaning **********
        df = self._prepare_data(self.df)

        # ********** Feature engineering **********
        df = self._feature_engineering(df)

        flag, report = DataValidator(
            df=df,
            date_col_name="ds",
            target_col_name="y",
            is_training_data=self.is_training_data,
            regressors_list=["HDD", "CDD"],
        ).validate_data()

        if self.is_training_data:
            return df[["ds", "y", "HDD", "CDD"]], flag, report
        else:
            return df[["ds", "HDD", "CDD"]], flag, report
