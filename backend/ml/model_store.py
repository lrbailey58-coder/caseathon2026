"""Save and load trained models, so predictions don't need retraining.

A saved file holds the two LightGBM models (north and east corrections) plus
everything needed to use them correctly: the feature list, the prediction
horizon, and where the training data came from.

Only load files you or your team created: loading a .joblib file can run code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib

from backend.ml.features import FEATURE_COLUMNS, HORIZON_S

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"  # git-ignored: rebuild models by re-running the scripts


def save_models(models: dict, name: str, info: dict | None = None) -> Path:
    """Save models to models/<name>.joblib and return the path."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(
        {
            "models": models,
            "feature_columns": FEATURE_COLUMNS,
            "horizon_s": HORIZON_S,
            "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "info": info or {},
        },
        path,
    )
    return path


def load_models(path: str | Path) -> dict:
    """Load a saved bundle. Refuses bundles built with a different feature list or horizon."""
    bundle = joblib.load(path)
    if bundle["feature_columns"] != FEATURE_COLUMNS or bundle["horizon_s"] != HORIZON_S:
        raise ValueError(
            f"{path} was trained with different features or horizon than the current code. Retrain it."
        )
    return bundle
