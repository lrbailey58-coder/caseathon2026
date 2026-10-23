"""The trust gate: the slow, accurate feed (TIER1_ANCHOR) audits the fast feed (TIER2_FAST).

The idea in one line: if a fast-feed ping disagrees with where the trusted feed
says the ship was, don't learn from it.

For every TIER2_FAST ping we look up the same voyage's TIER1_ANCHOR pings and ask
three questions:

    1. Position: how far (meters) is this ping from the anchor track at that moment?
    2. Speed:    how different (knots) is its speed from the anchor's speed?
    3. Heading:  how different (degrees) is its heading? (only checked when the
                 ship moves >= 3 knots; heading is meaningless for a nearly stopped ship)

"The anchor track at that moment" = straight-line interpolation between the two
anchor pings around it (anchors are ~6 min apart). If there is no anchor ping
before AND after within 20 min, we extend the nearest anchor (within 10 min) by
its own speed and heading. If no anchor is that close, the ping is "unverifiable"
and we keep it (we can't prove it is bad).

How far is "too far"? We LEARN normal limits from week W1, which is known to be
clean: the limit for each check is the 99th percentile of that check on W1's fast
feed. So ~1% of good pings exceed each limit by chance; that is the price of the gate.
Limits are learned PER SHIP TYPE, because "normal" differs a lot: a cargo ship
268 m off its track is wrong, but a maneuvering tug often is. A type with fewer
than 500 reference pings (e.g. tankers) uses the all-ships limits.

Trust score (0 to 1, shown to the watch officer):
    trust = min(1, limit / value) over the three checks
    1.0 = everything within normal limits; 0.5 = twice as far off as normal.
A ping is SUSPECT when trust < 1. TIER1_ANCHOR pings always get trust 1.

The gate never sees `is_injected_defect`; it only uses the two feeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.ml.features import vessel_group
from backend.ml.metrics import KNOTS_TO_MPS, haversine_m, offset_m_to_latlon

MAX_ANCHOR_GAP_S = 20 * 60  # interpolate only between anchors at most 20 min apart
MAX_EXTRAPOLATE_S = 10 * 60  # otherwise extend the nearest anchor by at most 10 min
MIN_SPEED_FOR_HEADING_KN = 3.0  # heading check only for ships moving at least this fast
LIMIT_PERCENTILE = 99.0  # "normal" = within the 99th percentile of the clean week W1
MIN_PINGS_FOR_OWN_LIMITS = 500  # ship types with fewer reference pings use the all-ships limits
CHECKS = ["anchor_dist_m", "speed_diff_kn", "heading_diff_deg"]


def _angle_diff(a, b):
    """Smallest difference between two headings in degrees (0-180)."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def compare_to_anchor(pings: pd.DataFrame) -> pd.DataFrame:
    """Add the three check values to every TIER2_FAST ping (NaN when unverifiable).

    `pings` = one condition, both feeds, contract columns (from data.load_pings).
    """
    df = pings.sort_values(["voyage_id", "timestamp"]).reset_index(drop=True)
    n = len(df)
    exp_lat, exp_lon = np.full(n, np.nan), np.full(n, np.nan)
    exp_sog, exp_cog = np.full(n, np.nan), np.full(n, np.nan)

    t_all = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().to_numpy()
    lat_all, lon_all = df["lat"].to_numpy(float), df["lon"].to_numpy(float)
    sog_all, cog_all = df["sog"].to_numpy(float), df["cog"].to_numpy(float)
    is_anchor = (df["source_tier"] == "TIER1_ANCHOR").to_numpy()

    for rows in df.groupby("voyage_id", sort=False).indices.values():
        a = rows[is_anchor[rows]]  # anchor rows of this voyage (time-sorted)
        f = rows[~is_anchor[rows]]  # fast-feed rows of this voyage
        if a.size == 0 or f.size == 0:
            continue
        at, ft = t_all[a], t_all[f]
        after = np.searchsorted(at, ft)  # index of the first anchor after each fast ping
        before = after - 1

        # Case 1: anchors on both sides, close together -> interpolate between them.
        has_both = (before >= 0) & (after < a.size)
        b_i, a_i = np.clip(before, 0, a.size - 1), np.clip(after, 0, a.size - 1)
        gap = at[a_i] - at[b_i]
        interp = has_both & (gap <= MAX_ANCHOR_GAP_S) & (gap > 0)
        w = np.where(interp, (ft - at[b_i]) / np.where(gap > 0, gap, 1), 0.0)  # 0 = at earlier anchor
        b_rows, a_rows = a[b_i], a[a_i]
        lat_i = lat_all[b_rows] + w * (lat_all[a_rows] - lat_all[b_rows])
        lon_i = lon_all[b_rows] + w * (lon_all[a_rows] - lon_all[b_rows])
        sog_i = sog_all[b_rows] + w * (sog_all[a_rows] - sog_all[b_rows])
        # headings are circular, so interpolate their sine and cosine
        rad_b, rad_a = np.radians(cog_all[b_rows]), np.radians(cog_all[a_rows])
        cog_i = np.degrees(np.arctan2((1 - w) * np.sin(rad_b) + w * np.sin(rad_a),
                                      (1 - w) * np.cos(rad_b) + w * np.cos(rad_a))) % 360

        # Case 2: otherwise extend the nearest anchor by its own speed and heading.
        dist_before = np.where(before >= 0, ft - at[b_i], np.inf)
        dist_after = np.where(after < a.size, at[a_i] - ft, np.inf)
        use_before = dist_before <= dist_after
        near = np.where(use_before, b_rows, a_rows)
        dt = ft - t_all[near]  # signed seconds from that anchor
        extrap = ~interp & (np.minimum(dist_before, dist_after) <= MAX_EXTRAPOLATE_S)
        move_m = sog_all[near] * KNOTS_TO_MPS * dt
        lat_e, lon_e = offset_m_to_latlon(lat_all[near], lon_all[near],
                                          move_m * np.cos(np.radians(cog_all[near])),
                                          move_m * np.sin(np.radians(cog_all[near])))

        exp_lat[f] = np.where(interp, lat_i, np.where(extrap, lat_e, np.nan))
        exp_lon[f] = np.where(interp, lon_i, np.where(extrap, lon_e, np.nan))
        exp_sog[f] = np.where(interp, sog_i, np.where(extrap, sog_all[near], np.nan))
        exp_cog[f] = np.where(interp, cog_i, np.where(extrap, cog_all[near], np.nan))

    df["anchor_dist_m"] = haversine_m(df["lat"], df["lon"], exp_lat, exp_lon)
    df["speed_diff_kn"] = np.abs(df["sog"].to_numpy(float) - exp_sog)
    heading_diff = _angle_diff(df["cog"], exp_cog)
    moving = (exp_sog >= MIN_SPEED_FOR_HEADING_KN) & (df["sog"].to_numpy(float) >= MIN_SPEED_FOR_HEADING_KN)
    df["heading_diff_deg"] = np.where(moving, heading_diff, np.nan)
    df["verifiable"] = ~is_anchor & ~np.isnan(exp_lat)
    df["vessel_group"] = vessel_group(df["vessel_type"])
    return df


def learn_limits(checked_w1: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Normal limits = 99th percentile of each check on the clean reference week's fast feed.

    Returns {"all": {...}, "0": {...}, ...}: all-ships limits plus one set per ship
    type (vessel_group code) that has enough reference pings.
    """
    fast = checked_w1[checked_w1["verifiable"]]
    if len(fast) < MIN_PINGS_FOR_OWN_LIMITS:
        raise ValueError(f"Only {len(fast)} verifiable reference pings; need >= {MIN_PINGS_FOR_OWN_LIMITS}.")
    pct = lambda d: {c: float(np.nanpercentile(d[c], LIMIT_PERCENTILE)) for c in CHECKS}
    limits = {"all": pct(fast)}
    for group, d in fast.groupby("vessel_group"):
        if len(d) >= MIN_PINGS_FOR_OWN_LIMITS:
            limits[str(int(group))] = pct(d)
    return limits


def limits_for_rows(df: pd.DataFrame, limits: dict[str, dict[str, float]]) -> dict[str, np.ndarray]:
    """Per-row limit for each check, using the row's ship-type limits when we have them."""
    keys = df["vessel_group"].astype(int).astype(str).to_numpy()
    keys = np.where(np.isin(keys, list(limits)), keys, "all")
    return {c: np.array([limits[k][c] for k in keys]) for c in CHECKS}


def score_trust(checked: pd.DataFrame, limits: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Add trust_score (0-1) and suspect (True/False). Anchors = 1.0; unverifiable = NaN (kept)."""
    df = checked.copy()
    row_limits = limits_for_rows(df, limits)
    ratios = [row_limits[c] / np.maximum(df[c].to_numpy(float), 1e-9) for c in CHECKS]  # >1 = within limit
    trust = np.nanmin(np.vstack([np.where(np.isnan(r), np.inf, r) for r in ratios]), axis=0)
    trust = np.minimum(trust, 1.0)
    df["trust_score"] = np.where(df["verifiable"], trust, np.nan)
    df.loc[df["source_tier"] == "TIER1_ANCHOR", "trust_score"] = 1.0
    df["suspect"] = df["trust_score"] < 1.0  # NaN (unverifiable) -> False -> kept
    return df


def apply_gate(pings: pd.DataFrame, limits: dict[str, dict[str, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept_pings, all_pings_with_scores). Kept = everything except suspect fast-feed pings."""
    scored = score_trust(compare_to_anchor(pings), limits)
    kept = scored.loc[~scored["suspect"], pings.columns].reset_index(drop=True)
    return kept, scored
