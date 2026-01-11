from datetime import datetime

import pandas as pd

from utils.logger import logger


class DataValidator:
    """
        Validate the structural, temporal, and business rule integrity of a time series dataset.

        This class performs a full validation pipeline required before training
        forecasting or regression models on time series data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the time series data.

        date_col_name : str
            Name of the column containing date values.

        target_col_name : str
            Name of the target variable column.

        is_training_data : bool
            Indicates if this dataset is used for training. Missing target values
            are allowed only if False.

        regressors_list : list of str, optional
            List of additional regressor column names. Default is None.

        business_rules : dict, optional
            Dictionary defining optional business constraints applied to the target
            and regressor variables. Default is None.

        flag : bool
            Global validation flag updated during validation checks.

        Raises
        ------
        KeyError
            If any of the required columns (`date_col_name`, `target_col_name`) are missing.

        Notes
        -----
        - Designed for MLOps pipelines: idempotent, fault-tolerant, and fully logged.
        - Logs all validation errors and warnings without interrupting execution.
        - Generates a structured report for monitoring and alerting.

        Example
        -------
        >>> import pandas as pd
        >>> from utils.validate_data import DataValidator
        >>> df = pd.DataFrame({
        ...     "date": pd.date_range("2026-01-01", periods=5),
        ...     "daily_conso_kwh": [10, 15, 14, 13, 12],
        ...     "temperature_mean": [5, 6, 4, 7, 5]
        ... })
        >>> validator = DataValidator(
        ...     df=df,
        ...     date_col_name="date",
        ...     target_col_name="daily_conso_kwh",
        ...     is_training_data=True,
        ...     regressors_list=["temperature_mean"]
        ... )
        >>> flag, report = validator.validate_data()
        >>> print(flag)
        True
        >>> print(report["missing_values"])
        {'date': 0, 'daily_conso_kwh': 0, 'temperature_mean': 0}
    """

    # ***** Initialization *****
    def __init__(
        self,
        df,
        date_col_name,
        target_col_name,
        is_training_data,
        regressors_list=None,
        business_rules=None,
    ):
        self.df = df
        self.date_col_name = date_col_name
        self.target_col_name = target_col_name
        self.is_training_data = is_training_data
        self.regressors_list = regressors_list
        self.business_rules = business_rules
        self.flag = True  # overall validation status

    # ***** Data Type Check *****
    def _check_data_type(self):
        """
        Validate data types for date, target, and regressor columns.
        """
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col_name]):
            logger.error(
                f"[DATA VALIDATION] Column '{self.date_col_name}' is not datetime-compatible"
            )
            self.flag = False

        # Check numeric columns
        non_numeric_features = [
            col
            for col in self.df.columns
            if col != self.date_col_name
            and not pd.api.types.is_numeric_dtype(self.df[col])
        ]
        if non_numeric_features:
            logger.error(
                f"[DATA VALIDATION] Non-numeric columns detected: {non_numeric_features}"
            )
            self.flag = False

    # ***** Missing Values Check *****
    def _check_missing_values(self):
        """
        Detect missing values in relevant columns depending on context
        (training vs forecasting).
        """
        df_cp = self.df.copy()
        if self.is_training_data:
            df_cp = df_cp.dropna(subset=[self.target_col_name])

        report = {col: df_cp[col].isnull().sum() for col in df_cp.columns}
        for key, value in report.items():
            if value != 0:
                logger.error(
                    f"[DATA VALIDATION] {value} missing values detected in {key}"
                )
                self.flag = False
        return report

    # ***** Duplicated Rows Check *****
    def _check_duplicated_rows(self):
        """
        Detect duplicated rows in the dataset.
        """
        duplicated_count = self.df.duplicated().sum()
        if duplicated_count != 0:
            logger.error(
                f"[DATA VALIDATION] {duplicated_count} duplicated rows detected"
            )
            self.flag = False

    # ***** Date Continuity & Order Check *****
    def _check_date_continuity_and_order(self):
        """
        Validate chronological order and continuity of the date column.
        """
        if not self.df[self.date_col_name].is_monotonic_increasing:
            logger.error("[DATA VALIDATION] Dates are not sorted chronologically")
            self.flag = False

        # Normalize and sort
        self.df = self.df.sort_values(self.date_col_name).reset_index(drop=True)
        self.df[self.date_col_name] = pd.to_datetime(
            self.df[self.date_col_name]
        ).dt.normalize()

        full_date_range = pd.date_range(
            self.df[self.date_col_name].min(),
            self.df[self.date_col_name].max(),
            freq="D",
        )
        existing_dates = self.df[self.date_col_name].unique()
        missing_dates = pd.DatetimeIndex(full_date_range).difference(existing_dates)
        if len(missing_dates) != 0:
            logger.error(
                f"[DATA VALIDATION] {len(missing_dates)} missing dates detected "
                f"between {missing_dates.min().date()} and {missing_dates.max().date()}"
            )
            self.flag = False

        return {
            "date_continuity_and_order": {
                "is_sorted": self.df[self.date_col_name].is_monotonic_increasing,
                "missing_dates": {
                    "count": len(missing_dates),
                    "range": (
                        f"{missing_dates.min().date()} to {missing_dates.max().date()}"
                        if len(missing_dates) > 0
                        else None
                    ),
                },
            }
        }

    # ***** Business Rules Check *****
    def _apply_business_rules(self):
        """
        Apply optional business rules to target and regressor columns.
        """
        report_section = {"target": {}, "regressors": {}}
        if not self.business_rules:
            return report_section

        # Target rules
        target_rules = self.business_rules.get("target", {})
        target_violations = {}
        if not target_rules.get("allow_negative", True):
            count = int((self.df[self.target_col_name] < 0).sum())
            if count > 0:
                target_violations["negative_values"] = count
        if "min_value" in target_rules:
            count = int(
                (self.df[self.target_col_name] < target_rules["min_value"]).sum()
            )
            if count > 0:
                target_violations["below_min"] = count
        if "max_value" in target_rules:
            count = int(
                (self.df[self.target_col_name] > target_rules["max_value"]).sum()
            )
            if count > 0:
                target_violations["above_max"] = count
        if target_violations:
            report_section["target"] = target_violations
            logger.warning(
                f"[DATA VALIDATION] Target rule violations: {target_violations}"
            )

        # Regressor rules
        regressor_rules = self.business_rules.get("regressors", {})
        for col, rules in regressor_rules.items():
            if col not in self.df.columns:
                continue
            col_violations = {}
            if "min_value" in rules:
                count = int((self.df[col] < rules["min_value"]).sum())
                if count > 0:
                    col_violations["below_min"] = count
            if "max_value" in rules:
                count = int((self.df[col] > rules["max_value"]).sum())
                if count > 0:
                    col_violations["above_max"] = count
            if col_violations:
                report_section["regressors"][col] = col_violations
                logger.warning(
                    f"[DATA VALIDATION] Rule violations for '{col}': {col_violations}"
                )

        return report_section

    # ***** Full Validation Pipeline *****
    def validate_data(self):
        """
        Run the complete validation pipeline.

        Returns
        -------
        bool
            Global validation flag.
        dict
            Detailed validation report including missing values, duplicates,
            date continuity, and business rule violations.
        """
        self._check_data_type()
        missing_values_report = self._check_missing_values()
        self._check_duplicated_rows()
        date_report = self._check_date_continuity_and_order()
        business_rules_report = self._apply_business_rules()

        report = {
            "date": datetime.utcnow().isoformat(),
            "data_types": {col: str(self.df[col].dtype) for col in self.df.columns},
            "missing_values": missing_values_report,
            "duplicated_rows": int(self.df.duplicated().sum()),
            "date_report": date_report,
            "business_rules": business_rules_report,
        }

        return self.flag, report
