import logging
import os
from logging.handlers import RotatingFileHandler

# --- Create logs directory if it doesn't exist ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "energy_forecast.log")

# --- Formatter ---
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Logger central ---
logger = logging.getLogger("energy_forecast_logger")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers
if not logger.handlers:

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler (rotatif) ---
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5_000_000,   # 5 MB
        backupCount=5         # garde 5 fichiers d'historique
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)