import os 
import pandas as pd 

from utils.logger import logger
from utils.validate_target_data import validate_target_data
from utils.paths import RAW_DIR, PROCESSED_DIR

def process_target(data_target_file_name):
    """
        Traite un fichier de données brutes et le prépare pour l'analyse.

        Cette fonction :
        1. Charge un fichier Parquet depuis le dossier `raw`.
        2. Copie le dataframe pour éviter de modifier l'original.
        3. Supprime la colonne debug éventuelle.
        4. Renomme les colonnes principales pour standardiser les noms.
        5. Convertit la colonne 'date' en datetime sans fuseau horaire.
        6. Sauvegarde le dataframe transformé dans le dossier `processed`.
        7. Logue le succès de l'opération via `utils.logger`.

        Paramètres :
        ----------
        data_target_file_name : str
            Nom du fichier Parquet à traiter (ex: 'bordeaux_conso_mwh.parquet').
        raw_data_folder_name : str, optionnel
            Nom du dossier contenant les données brutes (par défaut 'raw').
        process_data_folder_name : str, optionnel
            Nom du dossier où le fichier traité sera sauvegardé (par défaut 'processed').

        Exceptions :
        ----------
        FileNotFoundError
            Levée si le fichier source n'existe pas.
        RuntimeError
            Levée si une erreur inattendue survient lors de la lecture ou de la sauvegarde du fichier.

        Exemple :
        --------
        process_target("bordeaux_conso_mwh.parquet")
    """      
    # Construire le chemin complet vers le fichier source dans le dossier 'raw'
    path_for_loading = RAW_DIR / "bordeaux_conso_mwh.parquet"

    # Lecture du fichier Parquet avec gestion des erreurs
    try:
        df = pd.read_parquet(path_for_loading)  # Charge le fichier en dataframe
    except FileNotFoundError as e:
        # Si le fichier n'existe pas, lever une erreur explicite
        raise FileNotFoundError(
            f"Le fichier {path_for_loading} n'existe pas. Ingestion a dû échouer."
        ) from e
    except Exception as e:
        # Capturer toute autre erreur inattendue lors de la lecture
        raise RuntimeError(
            f"Erreur inattendue en lisant {path_for_loading}: {e}"
        ) from e

    # Copier le dataframe pour éviter de modifier l'original (protection du fichier source)
    df_cp = df.copy()

    # Supprimer la colonne de debug si elle existe (robuste grâce à errors='ignore')
    df_cp = df_cp.drop('range(date_heure,1day)', axis=1, errors='ignore')

    # Renommer les colonnes pour standardiser les noms et faciliter les traitements ultérieurs
    df_cp = df_cp.rename(columns={"jour": "date", "conso_jour": "daily_conso_mgw"})

    # Convertir la colonne 'date' en datetime, sans fuseau horaire (UTC→naive)
    df_cp["date"] = pd.to_datetime(df_cp["date"]).dt.tz_localize(None)
    df_cp = df_cp.sort_values("date")  # indispensable
    df_cp = df_cp.reset_index(drop=True)

    # Construire le chemin complet vers le fichier transformé dans le dossier 'processed'
    processed_path = PROCESSED_DIR / "bordeaux_conso_mwh.parquet"

    flag, report = validate_target_data(df_cp)

    if flag:
        # Sauvegarder le dataframe transformé et logger le succès
        try:
            df_cp.to_parquet(processed_path)  # Écrit le fichier Parquet transformé
            logger.info(f"[PROCESS] {data_target_file_name} - processed and saved")  # Log info
        except Exception as e:
            # Capturer toute erreur de sauvegarde et la remonter avec contexte
            raise RuntimeError(f"Erreur en sauvegardant {processed_path}: {e}") from e
    else:
        raise ValueError(
            f"Validation échouée pour {data_target_file_name}. "
            f"Voir report pour les détails : {report}"
        )
    
    return report

        

