import os 
import pandas as pd 

from utils.logger import logger


import pandas as pd
from utils.logger import logger

def validate_target_data(df):
    """
    Valide l’intégrité d’un dataset de consommation journalière.

    Cette fonction vérifie :
    - les valeurs manquantes par colonne,
    - les doublons par colonne,
    - les dates manquantes entre la date min et max du dataset,
    - l’ordre chronologique des dates,
    - les valeurs négatives et nulles dans la colonne 'daily_conso_mgw',
    - les valeurs aberrantes (> 4 sigma).

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant au moins les colonnes 'date' et 'daily_conso_mgw'.

    Returns
    -------
    flag : bool
        True si toutes les validations passent, False sinon.
    report : dict
        Contient des informations détaillées sur les anomalies détectées :
        - missing_values : dict par colonne
        - duplicates : dict par colonne
        - missing_dates : int
        - unsorted_dates : bool
        - negative_values : int
        - zero_values : int
        - outliers : int
    """
    flag = True

    # Vérification des valeurs manquantes et doublons
    for col in df.columns:
        if df[col].isnull().sum() != 0:
            logger.error(f"{col} contient {df[col].isnull().sum()} valeurs manquantes")
            flag = False
    
    if df.duplicated().sum() != 0:
        logger.error(f"{col} contient {df[col].duplicated().sum()} valeurs dupliquées")
        flag = False

    # Normalisation des dates (supprime fuseau horaire et time info)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Générer la séquence complète de dates
    plage_date = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    # Différence pour trouver les dates réellement manquantes
    existing_dates = pd.to_datetime(df["date"]).dt.normalize().unique()
    missing_dates = pd.DatetimeIndex(plage_date).difference(existing_dates)

    if len(missing_dates) != 0:
        logger.error(
            f"{len(missing_dates)} dates manquantes détectées entre "
            f"{missing_dates.min().date()} et {missing_dates.max().date()}. "
            f"Vérifiez le fichier source pour la continuité."
        )
        flag = False

    # Vérifier l'ordre des dates
    if not df["date"].is_monotonic_increasing:
        logger.error("Les dates ne sont pas triées")
        flag = False

    # Valeurs négatives et nulles
    if (df["daily_conso_mgw"] < 0).any():
        logger.error("Valeurs négatives détectées")
        flag = False
    if (df["daily_conso_mgw"] == 0).any():
        logger.warning("Valeurs nulles détectées")
        flag = False

    # Valeurs aberrantes (> 4 sigma)
    mean = df["daily_conso_mgw"].mean()
    std = df["daily_conso_mgw"].std()
    outliers = df[df["daily_conso_mgw"] > mean + 4*std]
    if not outliers.empty:
        logger.warning(f"{len(outliers)} valeurs anormalement élevées détectées")

    # Report complet pour monitoring
    report = {
        "missing_values": {col: df[col].isnull().sum() for col in df.columns},
        "duplicates": {col: df[col].duplicated().sum() for col in df.columns},
        "missing_dates": len(missing_dates),
        "unsorted_dates": not df["date"].is_monotonic_increasing,
        "negative_values": (df["daily_conso_mgw"] < 0).sum(),
        "zero_values": (df["daily_conso_mgw"] == 0).sum(),
        "outliers": len(outliers)
    }

    return flag, report

