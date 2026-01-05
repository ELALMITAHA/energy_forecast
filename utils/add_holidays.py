# ***** Add French Holidays and Bridge Days *****
import pandas as pd
import holidays


def get_fr_holidays():
    """
    Generate a DataFrame of French public holidays, bridge days, and selected school holidays.

    This function prepares a holiday DataFrame compatible with Prophet, including:
    - Official French public holidays.
    - Bridge days (Friday following a Thursday holiday).
    - Simplified set of school holidays.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'ds' : date of the holiday
        - 'holiday' : holiday name/type
        - 'lower_window' : integer, start of the holiday effect window
        - 'upper_window' : integer, end of the holiday effect window
    """
    # Define years of interest
    years = range(2022, 2026)
    fr_holidays = holidays.CountryHoliday("FR", years=years)

    # Official public holidays
    holidays_df = pd.DataFrame(
        [(date, name) for date, name in fr_holidays.items()], columns=["ds", "holiday"]
    )
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
    holidays_df["lower_window"] = -1
    holidays_df["upper_window"] = 1

    # ***** Add bridge days (Thursday holiday → Friday) *****
    bridge_dates = [
        pd.Timestamp(d) + pd.Timedelta(days=1)
        for d in fr_holidays.keys()
        if pd.Timestamp(d).weekday() == 3  # Thursday
    ]
    bridge_df = pd.DataFrame(
        {
            "ds": bridge_dates,
            "holiday": "bridge_day",
            "lower_window": 0,
            "upper_window": 0,
        }
    )
    holidays_df = pd.concat([holidays_df, bridge_df], ignore_index=True)

    # ***** Add simplified school holidays *****
    vacances_dates = pd.to_datetime(
        [
            "2023-02-18",
            "2023-03-06",
            "2023-04-15",
            "2023-05-01",
            "2023-07-08",
            "2023-09-03",
        ]
    )
    vacances_df = pd.DataFrame(
        {
            "ds": vacances_dates,
            "holiday": "school_holiday",
            "lower_window": 0,
            "upper_window": 0,
        }
    )
    holidays_df = pd.concat([holidays_df, vacances_df], ignore_index=True)

    return holidays_df
