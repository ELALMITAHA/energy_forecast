import optuna


class OptunaTuner:
    """
    Hyperparameter tuning class using Optuna for any trainer interface.

    This class orchestrates hyperparameter optimization for a given trainer
    by executing multiple trials and evaluating performance with a custom metric.

    Parameters
    ----------
    trainer_cls : class
        Class implementing the trainer interface with methods:
        - `suggest_params(trial)` → dict of hyperparameters
        - `build(params)` → initialize model with hyperparameters
        - `fit(df)` → train the model on dataframe
        - `predict(df)` → return predictions as DataFrame or Series

    train_df : pd.DataFrame
        Training dataset for fitting models.

    test_df : pd.DataFrame
        Test dataset for evaluating models.

    metric_fn : callable
        Function to compute a metric between true and predicted values.
        Example: mean_absolute_error.

    n_trials : int, optional
        Number of Optuna trials. Default is 50.

    direction : {"minimize", "maximize"}, optional
        Optimization direction. Default is "minimize".

    target : str, optional
        Name of the target column in test_df. Default is "y".

    Behavior
    --------
    - Uses Optuna to search for optimal hyperparameters.
    - Builds, trains, and evaluates a new trainer for each trial.
    - Returns the best model, best parameters, and best metric score.

    Methods
    -------
    _objective(trial)
        Defines the objective function for Optuna trials.

    run()
        Run the hyperparameter optimization and return the best model and results.

    Notes
    -----
    - Designed for MLOps pipelines: idempotent and reproducible.
    - Easily adaptable to any trainer implementing the required interface.
    """

    def __init__(
        self,
        model_interface_cls,
        train_df,
        test_df,
        metric_fn,
        n_trials=50,
        direction="minimize",
    ):
        self.model_interface_cls = model_interface_cls
        self.train_df = train_df
        self.test_df = test_df
        self.metric_fn = metric_fn
        self.n_trials = n_trials
        self.direction = direction

    # ***** Objective function for Optuna *****
    def _objective(self, trial):
        # create one interface instance per trial
        trainer = self.model_interface_cls()

        # suggest hyperparameters
        params = trainer.suggest_params(trial)

        # build & train
        trainer.build(params)
        trainer.fit(self.train_df)

        # predict & evaluate
        forecast = trainer.predict(self.test_df)
        y_pred = forecast[trainer.pred_col]
        y_true = self.test_df[trainer.target_col]

        score = self.metric_fn(y_true, y_pred)
        return score

    # ***** Run hyperparameter tuning *****
    def run(self):
        study = optuna.create_study(direction=self.direction)
        study.optimize(self._objective, n_trials=self.n_trials)

        best_params = study.best_params
        best_value = study.best_value

        # build & fit final model
        best_trainer = self.model_interface_cls()
        best_trainer.build(best_params)
        best_model = best_trainer.fit(self.train_df)

        return best_model, best_params, best_value
