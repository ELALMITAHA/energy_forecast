import datetime

# ************ COMMON DATE RANGES ************
# Shared date boundaries for consumption and weather data

# Today's date
today = datetime.datetime.today()

# Start date: same month & day as today, but in 2022
start_date = datetime.datetime(year=2022, month=today.month, day=today.day)
START_DATE = start_date.strftime("%Y-%m-%d")

# End date: today
END_DATE = today.strftime("%Y-%m-%d")
TODAY_UTC = END_DATE


# ************ OSD ELECTRICITY CONSUMPTION API ************
OSD_BASE_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/records"
)

OSD_SOURCES = {
    "bordeaux_conso_kwh": {
        "dataset": "eco2mix-metropoles-tr",
        "refine": {
            "libelle_metropole": "Bordeaux Métropole",
        },
    },
}

OSD_PARAMS = {
    "bordeaux_conso_kwh": {
        # Daily aggregated consumption (kWh)
        "select": "min(date_heure) as jour, sum(consommation) as conso_jour",
        "where": (
            f"date_heure >= '{START_DATE}T00:00:00' "
            f"AND date_heure < '{TODAY_UTC}T00:00:00'"
        ),
        "group_by": "range(date_heure, 1 day)",
        "order_by": "jour asc",
        "limit": 5000,
        "refine[libelle_metropole]": "Bordeaux Métropole",
    },
}


# ************ TEMPORARY DATA AVAILABILITY CAP ************
"""
IMPORTANT NOTE (MLOps decision):

Electricity consumption data (eco2mix) can be published with a delay of several days,
while weather data remains available up to the current day.

To avoid temporal misalignment between the target (consumption) and exogenous features
(weather) during training and evaluation, weather data is temporarily capped to the
last known availability date of the consumption data.

This cap is ONLY applied to historical data preparation.
Forecasting still relies on future weather data as expected.

This behavior is intentional and documented.
"""

# ⚠️ TEMPORARY: last known availability date of consumption data
# Update automatically once the API catches up
TARGET_DATA_AVAILABLE_UNTIL = "2026-01-08"


# ************ OPEN-METEO WEATHER API ************
OPEN_METEO_BASE_URL = {
    "bordeaux_weather_archive": "https://archive-api.open-meteo.com/v1/archive",
    "bordeaux_weather_forecast": "https://api.open-meteo.com/v1/forecast",
}

# --- Bordeaux geographical coordinates
LAT = 44.84
LON = -0.58

# --- Daily weather variables used as regressors
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
]

WEATHER_SOURCES = {
    "bordeaux_weather_archive": {
        "base": "archive",
        "lat": LAT,
        "lon": LON,
        "daily": ",".join(DAILY_VARS),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": "auto",
    },
    "bordeaux_weather_forecast": {
        "base": "forecast",
        "lat": LAT,
        "lon": LON,
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
    },
}
