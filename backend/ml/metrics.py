"""Distance helpers and the two error numbers we report.

Every error in this project is a *displacement error in meters*: how far the
predicted position (15 minutes ahead) is from where the ship really was.
We report two numbers, named to match the `experiment_metrics` table in
backend/db.py:

    displacement_mae_m -> average error in meters (the typical miss)
    tail_95_error_m    -> 95th-percentile error in meters (the bad misses)
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
M_PER_DEG_LAT = np.pi * EARTH_RADIUS_M / 180.0  # ~111,195 m in one degree of latitude
KNOTS_TO_MPS = 0.514444  # 1 knot = 0.514444 meters per second


def latlon_to_offset_m(lat0, lon0, lat1, lon1):
    """How far point 1 is from point 0, as (meters north, meters east).

    Uses a flat-Earth approximation. Over the ~10 km a ship moves in
    15 minutes this is accurate to within ~2 m, far below our model errors.
    """
    lat0 = np.asarray(lat0, dtype=float)
    lon0 = np.asarray(lon0, dtype=float)
    north_m = (np.asarray(lat1, dtype=float) - lat0) * M_PER_DEG_LAT
    east_m = (np.asarray(lon1, dtype=float) - lon0) * M_PER_DEG_LAT * np.cos(np.radians(lat0))
    return north_m, east_m


def offset_m_to_latlon(lat0, lon0, north_m, east_m):
    """Start at (lat0, lon0), move north_m and east_m meters, return the new (lat, lon).

    This is the exact inverse of latlon_to_offset_m.
    """
    lat0 = np.asarray(lat0, dtype=float)
    lon0 = np.asarray(lon0, dtype=float)
    lat1 = lat0 + np.asarray(north_m, dtype=float) / M_PER_DEG_LAT
    lon1 = lon0 + np.asarray(east_m, dtype=float) / (M_PER_DEG_LAT * np.cos(np.radians(lat0)))
    return lat1, lon1


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two sets of points (works on arrays)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def summarize_errors(errors_m) -> dict[str, float]:
    """Turn a list of per-prediction errors (meters) into the numbers we report."""
    errors_m = np.asarray(errors_m, dtype=float)
    if errors_m.size == 0:
        raise ValueError("No predictions to score.")
    if np.isnan(errors_m).any():
        raise ValueError("Some errors are NaN; check predictions and targets.")
    return {
        "displacement_mae_m": float(np.mean(errors_m)),
        "tail_95_error_m": float(np.percentile(errors_m, 95)),
        "n_predictions": int(errors_m.size),
    }
