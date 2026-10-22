"""Turn raw AIS pings into model inputs (features) and answers (targets).

For each ping we ask: "Where will this ship be 15 minutes from now?"

Features (inputs) use ONLY the current ping and the one before it, never the future:
    lat, lon            where the ship is now
    sog                 speed over ground (knots)
    cog_sin, cog_cos    heading, encoded so 359 and 1 degrees look close (they are)
    v_north_kn, v_east_kn  speed split into north and east parts
    dt_prev_s           seconds since the previous ping
    d_sog_per_min       how fast speed is changing (knots per minute)
    d_cog_per_min       how fast the ship is turning (degrees per minute)
    vessel_group        0 = cargo, 1 = tanker, 2 = tug/towing, 3 = other
                        (tugs move very differently from ships following the channel)

Targets (answers) come from the FUTURE and are only used for training and scoring:
    target_north_m, target_east_m   meters the ship moves north / east in 15 minutes
    target_lat, target_lon          the true position 15 minutes later

We predict movement in meters (instead of change in lat/lon degrees) because
a degree of longitude is a different number of meters at different latitudes,
and our error is reported in meters anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.ml.metrics import KNOTS_TO_MPS, latlon_to_offset_m

HORIZON_S = 15 * 60  # predict 15 minutes ahead
MAX_TARGET_GAP_S = 5 * 60  # skip a target if pings around t+15 are > 5 min apart (too uncertain)

FEATURE_COLUMNS = [
    "lat", "lon", "sog", "cog_sin", "cog_cos", "v_north_kn", "v_east_kn",
    "dt_prev_s", "d_sog_per_min", "d_cog_per_min", "vessel_group",
]
TARGET_COLUMNS = ["target_north_m", "target_east_m"]

# Columns the model must NEVER see. The defect flag and condition label are
# answers about the experiment itself; seeing them would be cheating (leakage).
FORBIDDEN_FEATURES = {"is_injected_defect", "condition", "target_lat", "target_lon", *TARGET_COLUMNS}
assert not set(FEATURE_COLUMNS) & FORBIDDEN_FEATURES, "A forbidden column is listed as a feature!"


def vessel_group(vessel_type: pd.Series) -> np.ndarray:
    """Map vessel type to 0 = cargo, 1 = tanker, 2 = tug/towing, 3 = other.

    Accepts text labels ("CARGO", "TANKER", "TUG") or the numeric AIS type codes
    used by MarineCadastre (70-79 cargo, 80-89 tanker, 31/32/52 towing and tugs).
    """
    text = vessel_type.astype(str).str.upper()
    code = pd.to_numeric(vessel_type, errors="coerce")
    groups = [
        (text.str.startswith("CARGO") | code.between(70, 79)).to_numpy(),
        (text.str.startswith("TANKER") | code.between(80, 89)).to_numpy(),
        (text.str.contains("TUG") | text.str.contains("TOW") | code.isin([31, 32, 52])).to_numpy(),
    ]
    return np.select(groups, [0, 1, 2], default=3).astype(float)


def build_features(pings: pd.DataFrame) -> pd.DataFrame:
    """Add feature columns. Uses only the current and previous ping of each voyage."""
    df = pings.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Order pings in time within each voyage; drop exact duplicate timestamps.
    df = (
        df.sort_values(["voyage_id", "timestamp"])
        .drop_duplicates(["voyage_id", "timestamp"], keep="first")
        .reset_index(drop=True)
    )

    cog_rad = np.radians(df["cog"].astype(float))
    df["cog_sin"] = np.sin(cog_rad)
    df["cog_cos"] = np.cos(cog_rad)
    df["v_north_kn"] = df["sog"] * df["cog_cos"]
    df["v_east_kn"] = df["sog"] * df["cog_sin"]
    df["vessel_group"] = vessel_group(df["vessel_type"])

    # Changes since the previous ping in the SAME voyage (first ping of a voyage gets NaN,
    # which LightGBM handles natively).
    by_voyage = df.groupby("voyage_id", sort=False)
    dt_s = by_voyage["timestamp"].diff().dt.total_seconds()
    df["dt_prev_s"] = dt_s
    df["d_sog_per_min"] = by_voyage["sog"].diff() / (dt_s / 60)
    turn_deg = (by_voyage["cog"].diff() + 180) % 360 - 180  # shortest turn, e.g. 350 -> 10 is +20
    df["d_cog_per_min"] = turn_deg / (dt_s / 60)
    return df


def add_targets(df: pd.DataFrame, horizon_s: int = HORIZON_S, max_gap_s: int = MAX_TARGET_GAP_S) -> pd.DataFrame:
    """Add the true position `horizon_s` seconds later and the movement in meters.

    Pings are irregular, so the exact t+15 min position is linearly interpolated
    between the two pings around it. Rows are dropped when the voyage ends before
    t+15 min or the pings around t+15 min are too far apart to trust.

    Note for later: in poisoned conditions, targets should come from the clean
    reference data so we score against the truth, not against corrupted pings.
    """
    df = df.copy()
    target_lat = np.full(len(df), np.nan)
    target_lon = np.full(len(df), np.nan)
    t_all = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().to_numpy()
    lat_all = df["lat"].to_numpy(dtype=float)
    lon_all = df["lon"].to_numpy(dtype=float)

    for rows in df.groupby("voyage_id", sort=False).indices.values():
        if rows.size < 2:
            continue
        t = t_all[rows]  # already sorted in time by build_features
        future_t = t + horizon_s
        has_future = future_t <= t[-1]
        after = np.clip(np.searchsorted(t, future_t), 1, t.size - 1)  # first ping at/after t+15
        gap_ok = (t[after] - t[after - 1]) <= max_gap_s
        ok = has_future & gap_ok
        target_lat[rows[ok]] = np.interp(future_t[ok], t, lat_all[rows])
        target_lon[rows[ok]] = np.interp(future_t[ok], t, lon_all[rows])

    df["target_lat"] = target_lat
    df["target_lon"] = target_lon
    df["target_north_m"], df["target_east_m"] = latlon_to_offset_m(df["lat"], df["lon"], target_lat, target_lon)
    return df.dropna(subset=["target_lat", "target_lon"]).reset_index(drop=True)


def dead_reckoning_offsets(df: pd.DataFrame, horizon_s: int = HORIZON_S) -> tuple[np.ndarray, np.ndarray]:
    """Physics guess: meters (north, east) moved if the ship keeps its current speed and heading.

    Uses only the current ping, so it is safe to use at prediction time.
    """
    distance_m = df["sog"].to_numpy(dtype=float) * KNOTS_TO_MPS * horizon_s
    return distance_m * df["cog_cos"].to_numpy(dtype=float), distance_m * df["cog_sin"].to_numpy(dtype=float)


def make_dataset(pings: pd.DataFrame) -> pd.DataFrame:
    """Raw pings -> one row per usable ping, with features and 15-minute targets."""
    return add_targets(build_features(pings))
