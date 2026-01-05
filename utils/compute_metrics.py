from sklearn.metrics import mean_absolute_error


# ***** Mean Absolute Scaled Error (MASE) *****
def mase_metric(y_true, y_pred, seasonality=7):
    """
    Compute the Mean Absolute Scaled Error (MASE) for time series predictions.

    MASE compares model error against a naive seasonal forecast.

    Parameters
    ----------
    y_true : pd.Series
        Actual observed values.
    y_pred : pd.Series
        Predicted values by the model.
    seasonality : int, optional
        Seasonal period for naive forecast (default=7).

    Returns
    -------
    float
        MASE value; lower values indicate better forecast accuracy.
    """
    # Convert to numpy arrays
    y_true, y_pred = y_true.values, y_pred.values

    # Naive forecast (seasonal lag)
    naive = y_true[:-seasonality]
    y_true_trimmed = y_true[seasonality:]

    # Compute MAE for naive baseline
    mae_naive = mean_absolute_error(y_true_trimmed, naive)

    # Compute MAE for the model predictions
    mae_model = mean_absolute_error(y_true_trimmed, y_pred[seasonality:])

    # Return scaled error
    return mae_model / mae_naive


# ***** Rolling Window Evaluation *****
def evaluate_rolling_window(y_true, y_pred, window_size=60, seasonality=7):
    """
    Evaluate predictions over the most recent rolling window.

    Computes both simple MAE and MASE over the specified window.

    Parameters
    ----------
    y_true : pd.Series
        Actual observed values.
    y_pred : pd.Series
        Predicted values by the model.
    window_size : int, optional
        Number of most recent observations to evaluate (default=60).
    seasonality : int, optional
        Seasonal period used for MASE calculation (default=7).

    Returns
    -------
    dict
        Dictionary with evaluation metrics:
        - 'window_size' : int, evaluated window length
        - 'baseline' : str, naive seasonal baseline
        - 'mae_window' : float, MAE over the window
        - 'mase_window' : float, MASE over the window
    """
    # Select the rolling window
    y_true_w = y_true.iloc[-window_size:]
    y_pred_w = y_pred.loc[y_true_w.index]

    # Compute MAE over the window
    mae_val = mean_absolute_error(y_true_w, y_pred_w)

    # Compute MASE over the window
    mase_val = mase_metric(y_true_w, y_pred_w, seasonality=seasonality)

    # Return structured report
    return {
        "window_size": window_size,
        "baseline": f"naive_{seasonality}_days",
        "mae_window": round(mae_val, 4),
        "mase_window": round(mase_val, 4),
    }
