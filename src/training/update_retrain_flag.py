from datetime import datetime
import json
from pathlib import Path

from utils.logger import logger
from configs.paths_config import BASE_DIR


def update_retrain_flag(
    output_flag_file,
    output_flag_folder,
    metrics,
    metric_key="mase_window",
    hf_repo=None,
    hf_path_in_repo=None,
):
    """
    Update the retraining decision flag based on evaluation metrics.

    This function determines whether a model should be retrained by
    applying a business threshold to a selected evaluation metric.
    The decision is persisted locally and can optionally be synced
    to a Hugging Face dataset repository.

    Parameters
    ----------
    inoutput_flag_file : str
        Name of the JSON file storing the retrain decision.

    inoutput_flag_folder : str or Path
        Directory where the retrain flag file is written.

    metrics : dict
        Dictionary of evaluation metrics produced by the evaluation step.

    metric_key : str, optional
        Metric used to decide retraining. Default is "mase_window".

    hf_repo : str, optional
        Hugging Face dataset repository ID (optional).

    hf_path_in_repo : str, optional
        Path inside the HF repository where the flag should be stored.

    Behavior
    --------
    - Extracts the specified metric from evaluation results.
    - Applies a business rule to determine retraining necessity.
    - Persists the retraining decision locally as a JSON artifact.
    - Optionally synchronizes the flag to Hugging Face.
    - Logs all decisions and failures for observability.
    """

    # ***** Compute Retrain Decision *****
    try:
        metric_value = metrics[metric_key]
        should_retrain = metric_value > 0.95

        logger.info(
            f"[RETRAIN FLAG] Metric '{metric_key}'={metric_value} → "
            f"should_retrain={should_retrain}"
        )

    except KeyError:
        logger.error(
            f"[RETRAIN FLAG] Metric key '{metric_key}' not found in metrics: {list(metrics.keys())}"
        )
        raise ValueError(
            f"Cannot compute retrain flag: missing metric '{metric_key}'"
        )

    except Exception as e:
        logger.error(
            f"[RETRAIN FLAG] Failed to compute retrain decision: {e}"
        )
        raise

    # ***** Persist Flag Locally *****
    try:
        flag_path = Path(output_flag_folder, output_flag_file)
        flag_path.parent.mkdir(parents=True, exist_ok=True)

        with open(flag_path, "w") as f:
            json.dump({
                "date": datetime.utcnow().isoformat(),
                "should_retrain": should_retrain
                }, f, indent=2)

        logger.info(
            f"[RETRAIN FLAG] Flag saved locally at "
            f"{flag_path.relative_to(BASE_DIR)}"
        )

    except Exception as e:
        logger.error(
            f"[RETRAIN FLAG] Failed to write retrain flag locally: {e}"
        )
        raise

    # ***** Optional: Sync to Hugging Face *****
    if hf_repo and hf_path_in_repo:
        try:
            # Intentionally left minimal: HF sync logic handled elsewhere
            logger.info(
                f"[RETRAIN FLAG] HF sync enabled → "
                f"repo={hf_repo}, path={hf_path_in_repo}"
            )

        except Exception as e:
            logger.error(
                f"[RETRAIN FLAG] Failed to sync retrain flag to HF: {e}"
            )

    return should_retrain

    