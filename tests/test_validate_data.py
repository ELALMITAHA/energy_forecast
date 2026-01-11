from pathlib import Path
import json
import pytest
import pandas as pd
from utils.validate_data import DataValidator

# -------------------------------
# _check_needed_cols tests
# -------------------------------


def test_check_needed_cols_pass():
    df = pd.DataFrame({"date": ["2025-01-01"], "target": [10], "temp": [20]})
    validator = DataValidator(
        df=df, date_col_name="date", target_col_name="target", regressors_list=["temp"]
    )

    validator._check_needed_cols()

    assert validator.flag is True


def test_check_needed_cols_fail():
    df = pd.DataFrame({"date": ["2025-01-01"], "target": [10]})
    validator = DataValidator(
        df=df,
        date_col_name="date",
        target_col_name="target",
        regressors_list=["temp"],  # missing
    )
    validator._check_needed_cols()

    assert validator.flag is False


# -------------------------------
# _check_data_type tests
# -------------------------------


def test_check_data_type_pass():
    df = pd.DataFrame({"date": ["2025-01-01"], "target": [10], "temp": [20]})
    df["date"] = pd.to_datetime(df["date"])
    validator = DataValidator(
        df=df, date_col_name="date", target_col_name="target", regressors_list=["temp"]
    )

    validator._check_data_type()

    assert validator.flag is True


def test_check_data_type_fail():
    df = pd.DataFrame({"date": ["2025-01-01"], "target": [10], "temp": ["abc"]})
    validator = DataValidator(
        df=df, date_col_name="date", target_col_name="target", regressors_list=["temp"]
    )

    validator._check_data_type()

    assert validator.flag is False


# -------------------------------
# _check_missing_values tests
# -------------------------------


def test_check_missing_values_pass():
    df = pd.DataFrame(
        {"date": ["2025-01-01", "2025-01-02"], "target": [10, 15], "temp": [20, 25]}
    )
    validator = DataValidator(
        df=df, date_col_name="date", target_col_name="target", regressors_list=["temp"]
    )
    validator._check_missing_values()

    assert validator.flag == True


def test_check_missing_values_fail():
    df = pd.DataFrame(
        {"date": ["2025-01-01", None], "target": [10, 15], "temp": [20, 25]}
    )
    validator = DataValidator(
        df=df, date_col_name="date", target_col_name="target", regressors_list=["temp"]
    )
    validator._check_missing_values()

    assert validator.flag == False


# -------------------------------
# _check_duplicated_rows tests
# -------------------------------


def test_check_duplicated_rows_pass():
    df = pd.DataFrame(
        {"date": ["2025-01-01", "2025-01-02"], "target": [10, 15], "temp": [20, 25]}
    )
    validator = DataValidator(
        df=df, date_col_name="date", target_col_name="target", regressors_list=["temp"]
    )
    validator._check_duplicated_rows()

    assert validator.flag == True


def test_check_duplicated_rows_fail():
    df = pd.DataFrame(
        {"date": ["2025-01-01", "2025-01-01"], "target": [10, 10], "temp": [20, 20]}
    )
    validator = DataValidator(
        df=df, date_col_name="date", target_col_name="target", regressors_list=["temp"]
    )
    validator._check_duplicated_rows()
    assert validator.flag == False


# -------------------------------
# _check_date_continuity_and_order tests
# -------------------------------


def test_check_dates_pass():
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "target": [10, 15, 20],
            "temp": [20, 25, 30],
        }
    )
    validator = DataValidator(
        df=df, date_col_name="date", target_col_name="target", regressors_list=["temp"]
    )
    df["date"] = pd.to_datetime(df["date"])
    validator._check_date_continuity_and_order()

    assert validator.flag == True


def test_check_dates_fail_missing():
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-03"],  # missing 2025-01-02
            "target": [10, 20],
            "temp": [20, 30],
        }
    )
    validator = DataValidator(
        df=df, date_col_name="date", target_col_name="target", regressors_list=["temp"]
    )
    df["date"] = pd.to_datetime(df["date"])
    validator._check_date_continuity_and_order()

    assert validator.flag == False


def test_check_dates_fail_unsorted():
    df = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-01"],  # unsorted
            "target": [15, 10],
            "temp": [25, 20],
        }
    )
    validator = DataValidator(
        df=df, date_col_name="date", target_col_name="target", regressors_list=["temp"]
    )
    df["date"] = pd.to_datetime(df["date"])
    validator._check_date_continuity_and_order()

    assert validator.flag == False


# -------------------------------
# Business rules tests
# -------------------------------


def test_business_rule_min_target_fail():
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "target": [10, -5],  # negative value
            "temp": [20, 25],
        }
    )
    validator = DataValidator(
        df=df,
        date_col_name="date",
        target_col_name="target",
        regressors_list=["temp"],
        business_rules={"target": {"allow_negative": False}},
    )
    report = validator._apply_business_rules()
    assert report["regressors"].get("temp", {}).get("below_min", 0) == 0


def test_business_rule_regressor_fail():
    df = pd.DataFrame(
        {"date": ["2025-01-01", "2025-01-02"], "target": [10, 15], "temp": [5, 25]}
    )
    validator = DataValidator(
        df=df,
        date_col_name="date",
        target_col_name="target",
        regressors_list=["temp"],
        business_rules={"regressors": {"temp": {"min_value": 10}}},
    )
    report = validator._apply_business_rules()

    # Nouvelle structure
    assert report == {"target": {}, "regressors": {"temp": {"below_min": 1}}}


# =========================================================
# Fixtures
# =========================================================


@pytest.fixture
def valid_dataframe():
    """
    Clean dataframe with no validation issues.
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "target": [100, 120, 110, 115, 130],
            "temp": [5.0, 6.0, 5.5, 6.2, 5.8],
        }
    )


@pytest.fixture
def missing_column_dataframe():
    """
    Dataframe missing the target column.
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="D"),
            "temp": [5.0, 6.0, 7.0],
        }
    )


@pytest.fixture
def invalid_dataframe():
    """
    Dataframe containing multiple validation issues:
    - missing date
    - duplicated row
    - missing value
    - negative target
    """
    df = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-04",  # missing 2025-01-03
                "2025-01-04",  # duplicate
            ],
            "target": [100, -20, 130, 130],
            "temp": [5.0, 6.0, None, 7.0],
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


# =========================================================
# Tests public methd :
# =========================================================


def test_validate_data_success(valid_dataframe):
    """
    Full validation should succeed on a clean dataset.
    """
    validator = DataValidator(
        df=valid_dataframe,
        date_col_name="date",
        target_col_name="target",
        regressors_list=["temp"],
    )

    flag, report = validator.validate_data()

    assert flag is True
    assert report["missing_columns"] == "No missing columns detected"
    assert report["duplicated_rows"] == 0
    assert all(v == 0 for v in report["missing_values"].values())


def test_validate_data_failure_missing_column(missing_column_dataframe):
    """
    Validation should fail if required columns are missing.
    """
    validator = DataValidator(
        df=missing_column_dataframe,
        date_col_name="date",
        target_col_name="target",
    )

    flag, report = validator.validate_data()

    assert flag is False
    assert report["missing_columns"] != "No missing columns detected"


def test_validate_data_failure_multiple_issues(invalid_dataframe):
    """
    Validation should fail when multiple issues are present.

    Checks performed:
    - missing value
    - negative target
    - missing date in the sequence
    - business rule violations
    - duplicated rows (strict: all columns identical)
    """
    validator = DataValidator(
        df=invalid_dataframe,
        date_col_name="date",
        target_col_name="target",
        regressors_list=["temp"],
        business_rules={
            "target": {
                "allow_negative": False,
                "min_value": 0,
                "max_value": 200,
            },
            "regressors": {
                "temp": {
                    "min_value": -30,
                    "max_value": 50,
                }
            },
        },
    )

    # Run validation
    flag, report = validator.validate_data()

    # Validation should fail
    assert flag is False

    # --------------------------
    # Structural checks
    # --------------------------
    # pandas duplicates: strict row equality
    assert report["duplicated_rows"] == 0

    # Missing values
    assert report["missing_values"]["temp"] == 1

    # --------------------------
    # Date continuity
    # --------------------------
    date_report = report["date_report"]["date_continuity_and_order"]

    # Accès à la nouvelle structure
    assert date_report["missing_dates"]["count"] == 1
    assert date_report["missing_dates"]["range"] == "2025-01-03 to 2025-01-03"

    # --------------------------
    # Business rules
    # --------------------------
    business_report = report["business_rules"]
    assert "target" in business_report
    assert business_report["target"]["negative_values"] == 1
    assert business_report["target"]["below_min"] == 1

    assert "regressors" in business_report
    assert business_report["regressors"].get("temp", {}).get("below_min", 0) == 0

    print(report)
