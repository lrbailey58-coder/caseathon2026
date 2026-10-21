"""Fake AIS pings shaped exactly like Person 1's real data.

Lets us build and test the ML code before the real data is ready.
Data contract agreed with Person 1 (one row per AIS ping):

    mmsi, voyage_id, timestamp (UTC), lat, lon, sog (knots), cog (degrees),
    vessel_type, window_id (1-6), source_tier (TIER1_ANCHOR / TIER2_FAST),
    condition, is_injected_defect

All dummy data is clean ("NOMINAL"). Corrupting data is the job of Person 1's
degradation injector. This is SYNTHETIC data: label it that way in any result.
Ships follow a simplified channel; positions are approximate and not checked against land.

Try it:  uv run python -m backend.ml.dummy_data
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backend.ml.metrics import KNOTS_TO_MPS, latlon_to_offset_m, offset_m_to_latlon

DATA_COLUMNS = [
    "mmsi", "voyage_id", "timestamp", "lat", "lon", "sog", "cog",
    "vessel_type", "window_id", "source_tier", "condition", "is_injected_defect",
]

FIRST_WEEK_START = pd.Timestamp("2026-01-05", tz="UTC")  # a Monday; window W1 starts here
N_WINDOWS = 6  # W1..W6, one week each
SIM_STEP_S = 10  # ship motion is simulated every 10 s, then pings are sampled from it
TIER1_SHARE = 0.20  # 20% of pings come from the slow, accurate feed (per Architecture.md)

# Simplified inbound channel into Norfolk (offshore -> Cape Henry -> Thimble Shoal ->
# Hampton Roads -> south to Norfolk). Big ships follow it, turning at the bends,
# which is the kind of location-based pattern the model can learn and physics cannot.
ROUTE_LATLON = np.array([
    (36.93, -75.90), (36.96, -76.02), (36.97, -76.12),
    (36.96, -76.25), (36.92, -76.33), (36.86, -76.33),
])
_REF_LAT, _REF_LON = ROUTE_LATLON[0]
ROUTE_M = np.column_stack(latlon_to_offset_m(_REF_LAT, _REF_LON, ROUTE_LATLON[:, 0], ROUTE_LATLON[:, 1]))

# How each vessel type moves. Cargo ships and tankers follow the channel.
# Tugs work locally with sharp turns: the valid maneuvers our gate may later
# wrongly flag as noise (the Requirement 7 "failure case").
VESSEL_PROFILES = {
    "CARGO": {"speed_kn": (10, 16), "max_turn_deg_s": 0.4},
    "TANKER": {"speed_kn": (8, 13), "max_turn_deg_s": 0.3},
    "TUG": {"speed_kn": (4, 10), "n_turns": (2, 4), "turn_deg": (60, 150), "turn_s": (30, 90)},
}
VESSEL_TYPE_PROBS = {"CARGO": 0.45, "TANKER": 0.35, "TUG": 0.20}
MAX_VOYAGE_S = 4 * 3600


def _smooth_noise(rng: np.random.Generator, n: int, scale: float, window: int = 30) -> np.ndarray:
    """Slowly-changing random noise (a moving average of white noise), std = scale."""
    white = rng.normal(0.0, 1.0, n + window)
    smooth = np.convolve(white, np.ones(window) / window, mode="valid")[:n]
    return smooth / smooth.std() * scale


def _bearing_deg(d_north: float, d_east: float) -> float:
    """Compass bearing (0 = north, 90 = east) of a movement."""
    return math.degrees(math.atan2(d_east, d_north)) % 360


def _right_of(direction: np.ndarray) -> np.ndarray:
    """Unit vector pointing to the right of a (north, east) direction of travel.

    Example: travelling north (1, 0) -> right is east (0, 1).
    """
    return np.array([-direction[1], direction[0]]) / np.hypot(direction[0], direction[1])


def _follow_channel(rng: np.random.Generator, p: dict) -> tuple[np.ndarray, ...]:
    """A big ship steering along the channel (inbound or outbound) on its own side of it."""
    inbound = rng.random() < 0.5
    pts = ROUTE_M if inbound else ROUTE_M[::-1]
    # Aim points sit 100-300 m to the right of the centerline (ships keep to starboard).
    lane_m = rng.uniform(100, 300)
    aims = [pts[i] + lane_m * _right_of(pts[i] - pts[i - 1]) for i in range(1, len(pts))]

    # Start at a random spot on one of the first legs, in our lane, pointing along the leg.
    leg_i = int(rng.integers(0, len(pts) - 2))
    leg = pts[leg_i + 1] - pts[leg_i]
    pos = pts[leg_i] + rng.uniform(0.0, 0.8) * leg + lane_m * _right_of(leg)
    north, east = float(pos[0]), float(pos[1])
    heading = _bearing_deg(*leg)

    n_max = MAX_VOYAGE_S // SIM_STEP_S
    speed_kn = np.clip(rng.uniform(*p["speed_kn"]) + _smooth_noise(rng, n_max, scale=0.8), 0.5, None)
    wobble = rng.normal(0.0, 0.3, n_max)  # helmsman / sea-state wobble, degrees per step
    max_turn = p["max_turn_deg_s"] * SIM_STEP_S

    out_n, out_e, out_h = [], [], []
    for step in range(n_max):
        out_n.append(north)
        out_e.append(east)
        out_h.append(heading)
        to_n, to_e = aims[leg_i][0] - north, aims[leg_i][1] - east
        if math.hypot(to_n, to_e) < 400:  # close to this bend: aim for the next one
            if leg_i == len(aims) - 1:
                break  # reached the end of the channel
            leg_i += 1
            to_n, to_e = aims[leg_i][0] - north, aims[leg_i][1] - east
        error = (_bearing_deg(to_n, to_e) - heading + 180) % 360 - 180
        heading = (heading + float(np.clip(error, -max_turn, max_turn)) + wobble[step]) % 360
        v = speed_kn[step] * KNOTS_TO_MPS * SIM_STEP_S
        north += v * math.cos(math.radians(heading))
        east += v * math.sin(math.radians(heading))

    n = len(out_n)
    return np.array(out_n), np.array(out_e), speed_kn[:n], np.array(out_h)


def _tug_work(rng: np.random.Generator, p: dict) -> tuple[np.ndarray, ...]:
    """A tug job near the channel (30-90 min): slow, with several sharp but valid turns."""
    n_steps = int(rng.uniform(0.5, 1.5) * 3600 // SIM_STEP_S) + 1
    turn_rate = _smooth_noise(rng, n_steps, scale=0.01)  # degrees per second
    for _ in range(rng.integers(p["n_turns"][0], p["n_turns"][1] + 1)):
        size_deg = rng.uniform(*p["turn_deg"]) * rng.choice([-1.0, 1.0])
        length = max(1, int(rng.uniform(*p["turn_s"]) / SIM_STEP_S))
        start = int(rng.integers(0, max(1, n_steps - length)))
        turn_rate[start:start + length] += size_deg / (length * SIM_STEP_S)
    heading = (rng.uniform(0, 360) + np.cumsum(turn_rate * SIM_STEP_S)) % 360
    speed_kn = np.clip(rng.uniform(*p["speed_kn"]) + _smooth_noise(rng, n_steps, scale=0.8), 0.5, None)

    v_mps = speed_kn * KNOTS_TO_MPS
    step_n = v_mps * np.cos(np.radians(heading)) * SIM_STEP_S
    step_e = v_mps * np.sin(np.radians(heading)) * SIM_STEP_S
    start_pt = ROUTE_M[int(rng.integers(len(ROUTE_M)))]
    north = start_pt[0] + np.concatenate([[0.0], np.cumsum(step_n)[:-1]])
    east = start_pt[1] + np.concatenate([[0.0], np.cumsum(step_e)[:-1]])
    return north, east, speed_kn, heading


def _simulate_voyage(rng: np.random.Generator, vessel_type: str) -> dict[str, np.ndarray]:
    """Simulate one voyage, then sample AIS pings from it with small sensor noise.

    Returns ping times (seconds since voyage start) and reported lat/lon/sog/cog.
    """
    p = VESSEL_PROFILES[vessel_type]
    if vessel_type == "TUG":
        north, east, speed_kn, heading = _tug_work(rng, p)
    else:
        north, east, speed_kn, heading = _follow_channel(rng, p)
    true_lat, true_lon = offset_m_to_latlon(_REF_LAT, _REF_LON, north, east)

    # AIS pings arrive irregularly, every 30-90 s (always > SIM_STEP_S, so no duplicates).
    last_t = (north.size - 1) * SIM_STEP_S
    gaps = rng.uniform(30, 90, size=int(last_t / 30) + 2)
    ping_t = np.concatenate([[0.0], np.cumsum(gaps)])
    ping_t = ping_t[ping_t <= last_t]
    k = np.round(ping_t / SIM_STEP_S).astype(int)  # which simulation step each ping reports

    # Small, realistic sensor noise on what the ship reports (~5 m GPS, 0.1 kn, 1 degree).
    n = k.size
    lat, lon = offset_m_to_latlon(true_lat[k], true_lon[k], rng.normal(0, 5, n), rng.normal(0, 5, n))
    sog = np.clip(speed_kn[k] + rng.normal(0, 0.1, n), 0.0, None)
    cog = (heading[k] + rng.normal(0, 1.0, n)) % 360
    return {"t_s": k * float(SIM_STEP_S), "lat": lat, "lon": lon, "sog": sog, "cog": cog}


def generate_dummy_ais(n_voyages_per_window: int = 80, n_vessels: int = 50, seed: int = 42) -> pd.DataFrame:
    """Build a clean, synthetic AIS dataset covering windows W1..W6.

    Same seed -> exactly the same data, so results are repeatable.
    Every voyage starts and ends inside one week, so no voyage crosses windows.
    """
    rng = np.random.default_rng(seed)
    mmsis = 366_000_000 + rng.choice(1_000_000, size=n_vessels, replace=False)
    types = rng.choice(list(VESSEL_TYPE_PROBS), size=n_vessels, p=list(VESSEL_TYPE_PROBS.values()))

    voyages = []
    for window_id in range(1, N_WINDOWS + 1):
        week_start = FIRST_WEEK_START + pd.Timedelta(weeks=window_id - 1)
        for _ in range(n_voyages_per_window):
            v = int(rng.integers(n_vessels))
            # Start early enough that even the longest voyage ends inside this week.
            start = week_start + pd.Timedelta(seconds=int(rng.uniform(0, 7 * 24 * 3600 - MAX_VOYAGE_S)))
            sim = _simulate_voyage(rng, str(types[v]))
            n = sim["t_s"].size
            voyages.append(pd.DataFrame({
                "mmsi": str(mmsis[v]),
                "voyage_id": f"{mmsis[v]}_{start:%Y%m%dT%H%M%S}",
                "timestamp": start + pd.to_timedelta(sim["t_s"], unit="s"),
                "lat": sim["lat"],
                "lon": sim["lon"],
                "sog": sim["sog"],
                "cog": sim["cog"],
                "vessel_type": str(types[v]),
                "window_id": window_id,
                "source_tier": np.where(rng.random(n) < TIER1_SHARE, "TIER1_ANCHOR", "TIER2_FAST"),
                "condition": "NOMINAL",
                "is_injected_defect": False,
            }))

    df = pd.concat(voyages, ignore_index=True)
    if df["voyage_id"].nunique() != N_WINDOWS * n_voyages_per_window:
        raise RuntimeError("Two voyages got the same voyage_id; change the seed.")
    return df.sort_values(["voyage_id", "timestamp"]).reset_index(drop=True)[DATA_COLUMNS]


if __name__ == "__main__":
    data = generate_dummy_ais()
    print(data.head(10).to_string())
    print(f"\n{len(data):,} pings, {data['voyage_id'].nunique()} voyages")
    print(data.groupby("window_id")["voyage_id"].nunique().rename("voyages per window").to_string())
