from src.training.base_trainer import BaseModel
from utils.add_holidays import get_fr_holidays
from prophet import Prophet


class ProphetInterface(BaseModel):
    """
        Prophet model interface for training and forecasting time series data.

        This class provides a standardized interface for:
        - Building a Prophet model with specified hyperparameters.
        - Suggesting hyperparameter search space for Optuna.
        - Fitting the model on training data.
        - Generating forecasts on test data, including regressors and holidays.

        Attributes
        ----------
        train_df : pd.DataFrame
            Training dataset.
        test_df : pd.DataFrame
            Testing dataset.
        model : Prophet
            Prophet model instance.
        target : str
            Name of the target column.
        regressors : list of str
            List of additional regressors used in the model.

        Raises
        ------
        ValueError
            If the input dataset is invalid or missing required columns.
        RuntimeError
            If model fitting or prediction fails unexpectedly.

        Notes
        -----
        - Designed for production pipelines with reproducibility.
        - Idempotent: repeated calls produce the same outputs if inputs are unchanged.
        - Structured logging is recommended for monitoring.

        Example
        -------
        >>> interface = ProphetInterface()
        >>> model = interface.build(params={"changepoint_prior_scale":0.005})
        >>> model = interface.fit(train_df=my_train_data)
        >>> forecast = interface.predict(test_df=my_test_data)
    """

    # ***** Initialization *****
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.model = None
        self.target = "y"
        self.regressors = ["HDD", "CDD"]

    # ***** Target and Prediction Columns *****
    @property
    def target_col(self):
        """Return the name of the target column."""
        return "y"

    @property
    def pred_col(self):
        """Return the name of the prediction column."""
        return "yhat"

    # ***** Build Model *****
    def build(self, params):
        """
        Build a Prophet model with specified hyperparameters.

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameters for the Prophet model.

        Returns
        -------
        Prophet
            Configured Prophet model instance.
        """
        self.model = Prophet(
            **params,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            holidays=get_fr_holidays(),
        )

        # Add regressors
        for reg in self.regressors:
            self.model.add_regressor(reg)

        return self.model

    # ***** Hyperparameter Suggestion for Optuna *****
    def suggest_params(self, trial):
        """
        Define the hyperparameter search space for Optuna tuning.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial object used to sample hyperparameters.

        Returns
        -------
        dict
            Suggested hyperparameters for Prophet.
        """
        return {
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale", 0.0005, 0.01, log=True
            ),
            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale", 1.0, 20.0
            ),
            "holidays_prior_scale": trial.suggest_int("holidays_prior_scale", 5, 50),
            "seasonality_mode": trial.suggest_categorical(
                "seasonality_mode", ["additive", "multiplicative"]
            ),
        }

    # ***** Fit Model *****
    def fit(self, train_df=None):
        """
        Fit the Prophet model on training data.

        Parameters
        ----------
        train_df : pd.DataFrame, optional
            Training dataset. If None, uses `self.train_df`.

        Returns
        -------
        Prophet
            Fitted Prophet model instance.
        """
        if train_df is not None:
            self.train_df = train_df

        self.model.fit(self.train_df)
        return self.model

    # ***** Predict / Forecast *****
    def predict(self, test_df=None):
        """
        Generate forecasts using the fitted Prophet model.

        Parameters
        ----------
        test_df : pd.DataFrame, optional
            Test dataset. If None, uses `self.test_df`.

        Returns
        -------
        pd.DataFrame
            Forecasted values including regressors and Prophet components.
        """
        if test_df is not None:
            self.test_df = test_df

        # Create future dataframe for forecasting
        future = self.model.make_future_dataframe(periods=len(self.test_df), freq="D")

        # Merge regressors
        future = future.merge(self.test_df[["ds"] + self.regressors], on="ds")

        # Generate forecasts
        forecast = self.model.predict(future)
        return forecast
