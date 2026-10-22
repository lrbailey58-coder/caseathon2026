""""I'm not sure" detection: hand uncertain predictions to the human watch officer.

Besides the main prediction, we train models that predict a RANGE: the 5th and
95th percentile of where the ship could be in 15 minutes (quantile regression,
one low and one high model for north and for east = 4 extra LightGBM models).
Like the main model, they learn corrections on top of the physics guess.

    uncertainty radius = distance from the center to the corner of that range box
                         (roughly: "the ship should be within this many meters")

If the radius is bigger than the SAFETY RADIUS, the prediction is DEFERRED: it is
still shown, but marked "needs human review" instead of being trusted automatically.

Safety radius = 1 nautical mile (1,852 m), a common closest-point-of-approach alarm
distance in collision avoidance. It is fixed in advance, NOT tuned on test results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.ml.features import FEATURE_COLUMNS, TARGET_COLUMNS, dead_reckoning_offsets

QUANTILES = (0.05, 0.95)  # a 90% range
SAFETY_RADIUS_M = 1852.0  # 1 nautical mile
DIRECTIONS = ("north", "east")


def train_uncertainty_models(train: pd.DataFrame, base_params: dict) -> dict[str, LGBMRegressor]:
    """Four models: low and high estimates of the north and east correction to physics."""
    physics = dict(zip(DIRECTIONS, dead_reckoning_offsets(train)))
    models = {}
    for direction, target in zip(DIRECTIONS, TARGET_COLUMNS):
        correction = train[target].to_numpy() - physics[direction]
        for q in QUANTILES:
            params = {**base_params, "objective": "quantile", "alpha": q}
            models[f"{direction}_q{int(q * 100):02d}"] = LGBMRegressor(**params).fit(train[FEATURE_COLUMNS], correction)
    return models


def predict_range(models: dict[str, LGBMRegressor], df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Low/high movement estimates in meters (physics + correction) for north and east."""
    physics = dict(zip(DIRECTIONS, dead_reckoning_offsets(df)))
    lo, hi = (f"q{int(q * 100):02d}" for q in QUANTILES)
    out = {}
    for d in DIRECTIONS:
        a = physics[d] + models[f"{d}_{lo}"].predict(df[FEATURE_COLUMNS])
        b = physics[d] + models[f"{d}_{hi}"].predict(df[FEATURE_COLUMNS])
        out[f"{d}_lo"], out[f"{d}_hi"] = np.minimum(a, b), np.maximum(a, b)  # guard against crossed quantiles
    return out


def uncertainty_radius_m(rng: dict[str, np.ndarray]) -> np.ndarray:
    """Half-diagonal of the 90% range box, in meters."""
    half_n = (rng["north_hi"] - rng["north_lo"]) / 2
    half_e = (rng["east_hi"] - rng["east_lo"]) / 2
    return np.hypot(half_n, half_e)


def evaluate_deferral(df: pd.DataFrame, errors_m: np.ndarray, rng: dict[str, np.ndarray],
                      safety_radius_m: float = SAFETY_RADIUS_M) -> dict:
    """How many predictions go to a human, and were they really the worse ones?"""
    radius = uncertainty_radius_m(rng)
    deferred = radius > safety_radius_m
    inside = lambda d, t: (t >= rng[f"{d}_lo"]) & (t <= rng[f"{d}_hi"])
    mean = lambda x: float(np.mean(x)) if x.size else None
    groups = df["vessel_group"].to_numpy()
    names = {0: "cargo", 1: "tanker", 2: "tug", 3: "other"}
    return {
        "safety_radius_m": safety_radius_m,
        "decisions_deferred_pct": round(float(100.0 * deferred.mean()), 2),
        "accepted_mae_m": mean(errors_m[~deferred]),
        "deferred_mae_m": mean(errors_m[deferred]),
        "accepted_tail_95_m": float(np.percentile(errors_m[~deferred], 95)) if (~deferred).any() else None,
        "median_radius_m": float(np.median(radius)),
        # Calibration: a 90% range should contain the true movement ~90% of the time.
        "range_coverage_north_pct": round(float(100.0 * inside("north", df["target_north_m"].to_numpy()).mean()), 1),
        "range_coverage_east_pct": round(float(100.0 * inside("east", df["target_east_m"].to_numpy()).mean()), 1),
        # Is the radius useful? Share of the worst 10% of predictions that got deferred.
        "worst_10pct_deferred_pct": round(
            float(100.0 * deferred[errors_m >= np.percentile(errors_m, 90)].mean()), 1),
        "deferred_pct_by_ship_type": {
            names[int(g)]: round(float(100.0 * deferred[groups == g].mean()), 1) for g in np.unique(groups)},
    }
