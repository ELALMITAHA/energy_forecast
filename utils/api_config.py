from pathlib import Path
from datetime import datetime

today_utc = datetime.utcnow().date()

BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/records"

OSD_SOURCES = {
    "bordeaux_conso_mwh" : {
        "dataset": "eco2mix-metropoles-tr",
        "refine": {"libelle_metropole": "Bordeaux Métropole"}
    },
    # Ajoute ici d'autres datasets si besoin
}

OSD_PARAMS = {
    "bordeaux_conso_mwh": {
        "select": "min(date_heure) as jour, sum(consommation) as conso_jour",
        "where":  f"date_heure >= '2022-01-01T00:00:00' AND date_heure < '{today_utc}T00:00:00'",
        "group_by": "range(date_heure, 1 day)",
        "order_by": "jour asc",
        "limit": 5000,
        "refine[libelle_metropole]": "Bordeaux Métropole"
    },
}

RAW_DIR = Path("data/raw")