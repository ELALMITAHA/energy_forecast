import logging
import os
from logging.handlers import RotatingFileHandler
import colorlog
from pathlib import Path

# --- Création du dossier logs si nécessaire ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "energy_forecast.log")

# ----- Formatter pour le fichier log (pas de couleur) -----
file_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# ----- Logger central -----
logger = logging.getLogger("energy_forecast_logger")
logger.setLevel(logging.INFO)

# Eviter les doublons de handlers
if not logger.handlers:

    # --- Console handler avec couleurs ---
    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s%(asctime)s [%(levelname)s] %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File handler (rotatif, sans couleur) ---
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5_000_000,  # 5 MB
        backupCount=5,  # garde 5 fichiers
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


# ====== Helper pour colorer les paths dans la console ======
def color_path(path: str, color_code: str = "94") -> str:
    """
    Colorie un path pour la console (ne touche pas au fichier log).

    Parameters
    ----------
    path : str
        Le chemin à colorer
    color_code : str, optional
        Code ANSI couleur (94=bleu, 95=magenta, 96=cyan, 93=jaune)
    """
    return f"\033[{color_code}m{path}\033[0m"


