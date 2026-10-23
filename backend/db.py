from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import duckdb

ConditionType = Literal["NOMINAL", "STALE", "LOW_QUALITY", "COMBINED"]
VesselPresetType = Literal["CARGO_CONTAINER", "HARBOR_TUG_RARE_MANEUVER"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DB_PATH = DATA_DIR / "maritime_case.duckdb"

# Real ML results (written by `uv run python -m backend.ml.export_dashboard`, committed to git).
# When this file exists, the backend loads it on start and uses it INSTEAD of the demo seed data
# below, so every copy of the site (local or hosted) shows real results.
REAL_RESULTS_PATH = REPO_ROOT / "results" / "dashboard_export_real.json"
METRIC_COLUMNS = [
    "condition", "mitigated", "window_id", "displacement_mae_m", "tail_95_error_m", "data_discarded_pct",
    "decisions_deferred_pct", "tier2_corrupted_volume_pct", "avg_data_staleness_min", "model_compute_latency_ms",
]
TRAJECTORY_COLUMNS = [
    "condition", "mitigated", "window_id", "mmsi", "vessel_name", "vessel_type", "ground_truth_json",
    "unmitigated_pred_json", "mitigated_pred_json", "uncertainty_poly_json", "tier1_divergence_m",
    "staleness_min", "trust_score",
]
# Extra per-ship fields from the real export (NULL when showing demo data).
EXAMPLE_COLUMNS = {
    "voyage_id": "VARCHAR",
    "example_mae_m": "DOUBLE",
    "example_tail_95_m": "DOUBLE",
    "example_discarded_pct": "DOUBLE",
    "example_deferred_pct": "DOUBLE",
    "max_turn_deg_per_min": "DOUBLE",
    "clean_false_alarm_pct": "DOUBLE",
    "explanation": "VARCHAR",
}
# What the site is currently showing; reported by /api/health.
DATA_SOURCE: dict[str, object] = {"kind": "demo", "detail": "built-in demo values (no real results file found)"}
_loaded_results_signature: tuple[float, int] | None = None

CONDITIONS: list[ConditionType] = ["NOMINAL", "STALE", "LOW_QUALITY", "COMBINED"]
VESSEL_PRESETS: list[VesselPresetType] = [
    "CARGO_CONTAINER",
    "HARBOR_TUG_RARE_MANEUVER",
]

CONDITION_LABELS: dict[ConditionType, str] = {
    "NOMINAL": "Reference / Nominal",
    "STALE": "Stale Only",
    "LOW_QUALITY": "Low-Quality Only",
    "COMBINED": "Combined Poisoning",
}

WINDOW_LABELS: dict[int, str] = {
    1: "W1: Baseline Reference",
    2: "W2: Early Feed Drift",
    3: "W3: Moderate Poisoning",
    4: "W4: Heavy Retraining Drift",
    5: "W5: Severe Channel Excursion",
    6: "W6: Unseen Test Period",
}

# Base ground-truth voyage along Chesapeake Bay / Thimble Shoal Channel into Port of Virginia (Norfolk)
BASE_GROUND_TRUTH_TRACK: list[tuple[float, float]] = [
    (36.9520, -75.9850),
    (36.9610, -76.0120),
    (36.9705, -76.0410),
    (36.9790, -76.0695),
    (36.9865, -76.0980),
    (36.9935, -76.1260),
    (36.9995, -76.1540),
    (37.0040, -76.1810),
    (37.0065, -76.2080),
    (37.0045, -76.2350),
    (36.9985, -76.2610),
    (36.9890, -76.2860),
]

# Requirement 7 Boundary / Rare-Event Ground-Truth Track:
# MMSI 367004890 - TUG CAPE HENRY executing a rapid legitimate emergency escort turn (COG delta > 45 deg/min)
TUG_RARE_MANEUVER_GROUND_TRUTH_TRACK: list[tuple[float, float]] = [
    (36.9520, -75.9850),
    (36.9610, -76.0120),
    (36.9705, -76.0410),
    (36.9788, -76.0680),
    (36.9852, -76.0910),
    (36.9775, -76.1060),  # High-G escort pivot begins (COG delta = 64.2 deg/min)
    (36.9665, -76.1180),
    (36.9540, -76.1275),
    (36.9420, -76.1355),
    (36.9310, -76.1420),
    (36.9210, -76.1475),
    (36.9120, -76.1520),
]

# Tangent continuation track where the Anchor Trust Gate falsely rejects Tier-2 turn updates
TUG_ANCHOR_LAG_TANGENT_TRACK: list[tuple[float, float]] = [
    (36.9520, -75.9850),
    (36.9610, -76.0120),
    (36.9705, -76.0410),
    (36.9788, -76.0680),
    (36.9858, -76.0935),
    (36.9922, -76.1190),  # Continues WNW along stale Tier-1 linear extrapolation
    (36.9980, -76.1445),
    (37.0028, -76.1695),
    (37.0055, -76.1940),
    (37.0040, -76.2180),
    (36.9990, -76.2410),
    (36.9920, -76.2630),
]

# Deterministic progression tables across rolling windows W1 -> W6
ERROR_PROGRESSION: dict[ConditionType, dict[str, list[float]]] = {
    "NOMINAL": {
        "unmitigated_mae": [42.0, 44.5, 43.8, 45.2, 46.1, 47.0],
        "mitigated_mae": [41.5, 42.8, 42.1, 43.0, 43.6, 44.2],
        "tail_95_unmitigated": [88.0, 92.0, 91.0, 94.5, 96.0, 98.0],
        "tail_95_mitigated": [85.0, 88.0, 87.5, 89.0, 90.5, 91.5],
        "corrupted_vol_pct": [2.0, 2.5, 2.2, 2.8, 3.0, 3.1],
        "staleness_min": [1.2, 1.3, 1.2, 1.4, 1.4, 1.5],
        "discarded_pct": [1.8, 2.1, 2.0, 2.4, 2.6, 2.8],
        "deferred_pct": [2.0, 2.0, 2.5, 2.5, 3.0, 3.0],
    },
    "STALE": {
        "unmitigated_mae": [43.0, 118.0, 215.0, 342.0, 468.0, 540.0],
        "mitigated_mae": [42.0, 56.0, 68.0, 79.0, 88.0, 94.0],
        "tail_95_unmitigated": [90.0, 245.0, 430.0, 685.0, 910.0, 1060.0],
        "tail_95_mitigated": [86.0, 115.0, 138.0, 162.0, 179.0, 192.0],
        "corrupted_vol_pct": [4.0, 18.0, 34.0, 51.0, 66.0, 72.0],
        "staleness_min": [1.5, 6.8, 12.4, 17.5, 22.8, 25.4],
        "discarded_pct": [3.2, 16.5, 31.0, 47.5, 61.2, 67.8],
        "deferred_pct": [2.5, 6.0, 10.5, 14.0, 18.5, 21.0],
    },
    "LOW_QUALITY": {
        "unmitigated_mae": [44.0, 134.0, 248.0, 389.0, 512.0, 595.0],
        "mitigated_mae": [42.5, 59.0, 74.0, 86.0, 96.0, 105.0],
        "tail_95_unmitigated": [92.0, 290.0, 520.0, 790.0, 1045.0, 1210.0],
        "tail_95_mitigated": [88.0, 124.0, 152.0, 178.0, 198.0, 215.0],
        "corrupted_vol_pct": [5.0, 20.0, 38.0, 55.0, 68.0, 75.0],
        "staleness_min": [1.4, 2.1, 2.8, 3.4, 3.9, 4.2],
        "discarded_pct": [4.1, 18.2, 35.4, 51.0, 63.8, 70.2],
        "deferred_pct": [3.0, 7.5, 12.0, 16.5, 20.0, 23.5],
    },
    "COMBINED": {
        "unmitigated_mae": [45.0, 165.0, 310.0, 495.0, 670.0, 785.0],
        "mitigated_mae": [43.0, 64.0, 82.0, 98.0, 112.0, 124.0],
        "tail_95_unmitigated": [95.0, 355.0, 640.0, 980.0, 1320.0, 1540.0],
        "tail_95_mitigated": [89.0, 134.0, 170.0, 204.0, 230.0, 252.0],
        "corrupted_vol_pct": [6.0, 24.0, 44.0, 62.0, 76.0, 82.0],
        "staleness_min": [1.6, 8.2, 14.8, 19.6, 24.5, 27.8],
        "discarded_pct": [5.0, 21.8, 40.5, 57.4, 71.0, 76.5],
        "deferred_pct": [3.5, 9.0, 15.0, 21.5, 26.0, 29.5],
    },
}


def build_tracks(
    condition: ConditionType,
    window_id: int,
    mitigated: bool,
    vessel_type: VesselPresetType = "CARGO_CONTAINER",
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    """Generate deterministic ground truth, unmitigated drift, mitigated recovery, and 95% cone."""
    w_idx = window_id - 1  # 0..5
    drift_scale = 0.0 if condition == "NOMINAL" else (w_idx / 5.0)

    if vessel_type == "HARBOR_TUG_RARE_MANEUVER":
        # Requirement 7 Boundary / Failure Case:
        # TUG CAPE HENRY executes a legitimate high-G emergency escort turn (COG delta = 64.2 deg/min).
        # The kinematic + Tier-1 Anchor Trust Gate falsely flags the valid rapid turn as a "Low-Quality / Sensor Spike"
        # and discards the high-frequency Tier-2 turn updates, causing the mitigated model to lag along the old
        # Tier-1 anchor heading until the Watch Officer intervenes.
        unmitigated_track: list[tuple[float, float]] = []
        mitigated_track: list[tuple[float, float]] = []
        upper_bound: list[tuple[float, float]] = []
        lower_bound: list[tuple[float, float]] = []

        for i, (gt_lat, gt_lon) in enumerate(TUG_RARE_MANEUVER_GROUND_TRUTH_TRACK):
            progress = max(0.0, (i - 2) / (len(TUG_RARE_MANEUVER_GROUND_TRUTH_TRACK) - 3))
            if i <= 2:
                unmitigated_track.append((gt_lat, gt_lon))
                mitigated_track.append((gt_lat, gt_lon))
                continue

            # Unmitigated model ingests the raw high-frequency Tier-2 turn pings (plus any condition drift)
            if condition == "NOMINAL":
                d_lat = -0.0008 * progress
                d_lon = 0.0006 * progress
            elif condition == "STALE":
                d_lat = 0.0095 * drift_scale * (progress**1.2)
                d_lon = 0.0080 * drift_scale * progress
            elif condition == "LOW_QUALITY":
                osc = 0.0028 if i % 2 == 0 else -0.0022
                d_lat = osc * max(0.35, drift_scale)
                d_lon = 0.0055 * drift_scale * progress
            else:  # COMBINED
                osc = 0.0032 if i % 2 == 0 else -0.0024
                d_lat = (0.0110 * drift_scale * progress) + osc * max(0.4, drift_scale)
                d_lon = 0.0095 * drift_scale * progress

            unmit_lat = round(gt_lat + d_lat, 6)
            unmit_lon = round(gt_lon + d_lon, 6)
            unmitigated_track.append((unmit_lat, unmit_lon))

            # Mitigated model suffers from false-positive Trust Gate rejection of the sharp turn:
            # it discards the valid high-frequency Tier-2 turn updates and lags along the pre-turn Tier-1 Anchor tangent
            lag_lat, lag_lon = TUG_ANCHOR_LAG_TANGENT_TRACK[i]
            mit_lat = round(lag_lat - 0.0012 * progress, 6)
            mit_lon = round(lag_lon + 0.0008 * progress, 6)
            mitigated_track.append((mit_lat, mit_lon))

            center_lat = mit_lat if mitigated else unmit_lat
            center_lon = mit_lon if mitigated else unmit_lon
            cone_half_width = (0.0024 + 0.0028 * max(0.4, drift_scale)) * (0.5 + 0.9 * progress)

            upper_bound.append((round(center_lat + cone_half_width, 6), round(center_lon, 6)))
            lower_bound.append((round(center_lat - cone_half_width, 6), round(center_lon, 6)))

        anchor_start = TUG_RARE_MANEUVER_GROUND_TRUTH_TRACK[2]
        uncertainty_polygon = [anchor_start] + upper_bound + list(reversed(lower_bound))
        return (
            TUG_RARE_MANEUVER_GROUND_TRUTH_TRACK,
            unmitigated_track,
            mitigated_track,
            uncertainty_polygon,
        )

    unmitigated_track = []
    mitigated_track = []
    upper_bound = []
    lower_bound = []

    for i, (lat, lon) in enumerate(BASE_GROUND_TRUTH_TRACK):
        progress = max(0.0, (i - 2) / (len(BASE_GROUND_TRUTH_TRACK) - 3))
        if i <= 2:
            unmitigated_track.append((lat, lon))
            mitigated_track.append((lat, lon))
            continue

        if condition == "NOMINAL":
            d_lat = 0.0005 * progress
            d_lon = 0.0004 * progress
        elif condition == "STALE":
            d_lat = 0.0145 * drift_scale * (progress**1.35)
            d_lon = 0.0180 * drift_scale * (progress**1.25)
        elif condition == "LOW_QUALITY":
            osc = 0.0035 if i % 2 == 0 else -0.0022
            d_lat = (-0.0155 * (progress**1.3) + osc) * drift_scale
            d_lon = 0.0110 * drift_scale * progress
        else:  # COMBINED
            osc = 0.0030 if i % 2 == 0 else -0.0015
            d_lat = (0.0195 * (progress**1.35) + osc) * drift_scale
            d_lon = 0.0225 * drift_scale * (progress**1.25)

        unmit_lat = round(lat + d_lat, 6)
        unmit_lon = round(lon + d_lon, 6)
        unmitigated_track.append((unmit_lat, unmit_lon))

        mit_lat = round(lat + d_lat * 0.14, 6)
        mit_lon = round(lon + d_lon * 0.14, 6)
        mitigated_track.append((mit_lat, mit_lon))

        center_lat = mit_lat if mitigated else unmit_lat
        center_lon = mit_lon if mitigated else unmit_lon
        cone_half_width = (0.0014 + 0.0022 * drift_scale) * (0.45 + 0.85 * progress)
        if not mitigated and condition != "NOMINAL":
            cone_half_width *= 1.65

        upper_bound.append((round(center_lat + cone_half_width, 6), round(center_lon, 6)))
        lower_bound.append((round(center_lat - cone_half_width, 6), round(center_lon, 6)))

    anchor_start = BASE_GROUND_TRUTH_TRACK[2]
    uncertainty_polygon = [anchor_start] + upper_bound + list(reversed(lower_bound))

    return BASE_GROUND_TRUTH_TRACK, unmitigated_track, mitigated_track, uncertainty_polygon


def build_explanation(
    condition: ConditionType,
    window_id: int,
    staleness_min: float,
    corrupted_pct: float,
    divergence_m: float,
    vessel_type: VesselPresetType = "CARGO_CONTAINER",
    discarded_pct: float | None = None,
) -> str:
    window_tag = "W6 (Unseen Test Period)" if window_id == 6 else f"W{window_id}"
    filtered_str = f"{discarded_pct:.1f}%" if discarded_pct is not None else "82.4%"
    if vessel_type == "HARBOR_TUG_RARE_MANEUVER":
        return (
            f"[{window_tag} // FAILURE MODE] TUG CAPE HENRY (MMSI 367004890) executes a legitimate high-G "
            "emergency escort turn in Norfolk channel (COG delta = 64.2°/min > 45°/min gate threshold). "
            f"Kinematic + Tier-1 Anchor Trust Gate falsely flags {filtered_str} of valid high-frequency Tier-2 turn pings as "
            f"'Low-Quality / Sensor Spike', causing the mitigated trajectory to lag {divergence_m:.0f}m along the "
            "pre-turn anchor vector until Watch Officer manual waypoint verification."
        )
    if condition == "NOMINAL":
        return (
            f"[{window_tag}] Tier-2 commercial stream aligns within 45m of delayed Tier-1 USCG coastal anchor ({filtered_str} filtered). "
            "Kinematic velocity and heading deltas remain within nominal Norfolk channel bounds."
        )
    if condition == "STALE":
        return (
            f"[{window_tag}] Tier-2 Fast Feed repeats cached vessel coordinates (avg age {staleness_min:.1f} min) "
            f"with fresh timestamps, deviating {divergence_m:.0f}m from Tier-1 USCG Anchor trajectory; Trust Gate filters {filtered_str} of Tier-2 records."
        )
    if condition == "LOW_QUALITY":
        return (
            f"[{window_tag}] Tier-2 Fast Feed exhibits severe GPS multipath jitter and SOG/COG spikes across "
            f"{corrupted_pct:.0f}% of retraining buffer ({filtered_str} filtered by Trust Gate); deviates {divergence_m:.0f}m from Tier-1 Anchor."
        )
    return (
        f"[{window_tag}] Combined poisoning: Tier-2 Fast Feed has {staleness_min:.1f} min staleness and "
        f"{corrupted_pct:.0f}% corrupted records ({filtered_str} filtered by Trust Gate), pulling unmitigated t+15 prediction {divergence_m:.0f}m "
        "outside Thimble Shoal inbound channel."
    )


def get_connection() -> duckdb.DuckDBPyConnection:
    """Connect to the local DuckDB analytical store, ensuring the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def load_export_rows(conn: duckdb.DuckDBPyConnection, export: dict) -> None:
    """Replace experiment_metrics and vessel_trajectories with the rows of a real export, in one transaction."""
    metrics, trajectories = export["experiment_metrics"], export["vessel_trajectories"]
    presets = {t["vessel_type"] for t in trajectories}
    if len(metrics) != 48 or not set(VESSEL_PRESETS) <= presets:
        raise ValueError(f"unexpected export: {len(metrics)} metric rows, presets {sorted(presets)}")
    traj_cols = TRAJECTORY_COLUMNS + list(EXAMPLE_COLUMNS)
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("DELETE FROM experiment_metrics")
        conn.execute("DELETE FROM vessel_trajectories")
        conn.executemany(
            f"INSERT INTO experiment_metrics ({', '.join(METRIC_COLUMNS)}) VALUES ({', '.join('?' * len(METRIC_COLUMNS))})",
            [[m[c] for c in METRIC_COLUMNS] for m in metrics],
        )
        conn.executemany(
            f"INSERT INTO vessel_trajectories ({', '.join(traj_cols)}) VALUES ({', '.join('?' * len(traj_cols))})",
            [[json.dumps(t[c]) if c.endswith("_json") else t.get(c) for c in traj_cols] for t in trajectories],
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def load_real_results(conn: duckdb.DuckDBPyConnection) -> bool:
    """Load REAL_RESULTS_PATH if it exists; reload only when the file changes. True = real data is live."""
    global _loaded_results_signature
    if not REAL_RESULTS_PATH.exists():
        return False
    stat = REAL_RESULTS_PATH.stat()
    signature = (stat.st_mtime, stat.st_size)
    if signature != _loaded_results_signature:
        try:
            export = json.loads(REAL_RESULTS_PATH.read_text())
            load_export_rows(conn, export)
        except Exception as exc:  # a broken file must not take the site down: report it, fall back to demo
            DATA_SOURCE.update(kind="demo", detail=f"could not load {REAL_RESULTS_PATH.name}: {exc}")
            print(f"[db] WARNING: {DATA_SOURCE['detail']}")
            return False
        _loaded_results_signature = signature
        DATA_SOURCE.update(
            kind="real",
            detail=f"{REAL_RESULTS_PATH.relative_to(REPO_ROOT).as_posix()} ({export.get('source', '')})",
            created_utc=export.get("created_utc"),
        )
    return True


def init_and_seed_db() -> None:
    """Create the core DuckDB tables if they do not exist and seed them when empty."""
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_metrics (
                condition VARCHAR,
                mitigated BOOLEAN,
                window_id INTEGER,
                displacement_mae_m DOUBLE,
                tail_95_error_m DOUBLE,
                data_discarded_pct DOUBLE,
                decisions_deferred_pct DOUBLE,
                tier2_corrupted_volume_pct DOUBLE,
                avg_data_staleness_min DOUBLE,
                model_compute_latency_ms DOUBLE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vessel_trajectories (
                condition VARCHAR,
                mitigated BOOLEAN,
                window_id INTEGER,
                mmsi VARCHAR,
                vessel_name VARCHAR,
                vessel_type VARCHAR,
                ground_truth_json VARCHAR,
                unmitigated_pred_json VARCHAR,
                mitigated_pred_json VARCHAR,
                uncertainty_poly_json VARCHAR,
                tier1_divergence_m DOUBLE,
                staleness_min DOUBLE,
                trust_score DOUBLE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hitl_audit_log (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                mmsi VARCHAR,
                vessel_name VARCHAR,
                condition VARCHAR,
                window_id INTEGER,
                action_taken VARCHAR,
                review_duration_seconds DOUBLE,
                operator_notes VARCHAR
            )
            """
        )

        for column, sql_type in EXAMPLE_COLUMNS.items():
            conn.execute(f"ALTER TABLE vessel_trajectories ADD COLUMN IF NOT EXISTS {column} {sql_type}")

        # Real results replace everything below; the demo seeds only run without them.
        real = load_real_results(conn)

        # Seed experiment_metrics if empty
        metrics_count_row = conn.execute("SELECT COUNT(*) FROM experiment_metrics").fetchone()
        metrics_count = int(metrics_count_row[0]) if metrics_count_row else 0
        if metrics_count == 0:
            metrics_rows: list[tuple[object, ...]] = []
            for condition in CONDITIONS:
                table = ERROR_PROGRESSION[condition]
                for mitigated in (False, True):
                    for window_id in range(1, 7):
                        w_idx = window_id - 1
                        mae = (
                            table["mitigated_mae"][w_idx]
                            if mitigated
                            else table["unmitigated_mae"][w_idx]
                        )
                        tail_95 = (
                            table["tail_95_mitigated"][w_idx]
                            if mitigated
                            else table["tail_95_unmitigated"][w_idx]
                        )
                        discarded_pct = table["discarded_pct"][w_idx] if mitigated else 0.0
                        deferred_pct = table["deferred_pct"][w_idx] if mitigated else 0.0
                        corrupted_pct = table["corrupted_vol_pct"][w_idx]
                        staleness_min = table["staleness_min"][w_idx]
                        latency_ms = round(11.8 + (1.6 if mitigated else 0.0) + 0.2 * w_idx, 1)

                        metrics_rows.append(
                            (
                                condition,
                                mitigated,
                                window_id,
                                mae,
                                tail_95,
                                discarded_pct,
                                deferred_pct,
                                corrupted_pct,
                                staleness_min,
                                latency_ms,
                            )
                        )

            conn.executemany(
                """
                INSERT INTO experiment_metrics (
                    condition,
                    mitigated,
                    window_id,
                    displacement_mae_m,
                    tail_95_error_m,
                    data_discarded_pct,
                    decisions_deferred_pct,
                    tier2_corrupted_volume_pct,
                    avg_data_staleness_min,
                    model_compute_latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                metrics_rows,
            )

        # Seed vessel_trajectories if empty
        traj_count_row = conn.execute("SELECT COUNT(*) FROM vessel_trajectories").fetchone()
        traj_count = int(traj_count_row[0]) if traj_count_row else 0
        if traj_count == 0:
            traj_rows: list[tuple[object, ...]] = []
            for condition in CONDITIONS:
                table = ERROR_PROGRESSION[condition]
                for mitigated in (False, True):
                    for window_id in range(1, 7):
                        w_idx = window_id - 1
                        gt_track, unmit_track, mit_track, cone_poly = build_tracks(
                            condition, window_id, mitigated, "CARGO_CONTAINER"
                        )
                        staleness_min = table["staleness_min"][w_idx]
                        corrupted_pct = table["corrupted_vol_pct"][w_idx]
                        divergence_m = round(table["unmitigated_mae"][w_idx] * 0.88, 1)
                        trust_score = round(max(0.12, 0.98 - (corrupted_pct / 100.0) * 0.85), 2)

                        traj_rows.append(
                            (
                                condition,
                                mitigated,
                                window_id,
                                "367591240",
                                "MV VIRGINIA TRADER",
                                "CARGO_CONTAINER",
                                json.dumps(gt_track),
                                json.dumps(unmit_track),
                                json.dumps(mit_track),
                                json.dumps(cone_poly),
                                divergence_m,
                                staleness_min,
                                trust_score,
                            )
                        )

            conn.executemany(
                """
                INSERT INTO vessel_trajectories (
                    condition,
                    mitigated,
                    window_id,
                    mmsi,
                    vessel_name,
                    vessel_type,
                    ground_truth_json,
                    unmitigated_pred_json,
                    mitigated_pred_json,
                    uncertainty_poly_json,
                    tier1_divergence_m,
                    staleness_min,
                    trust_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                traj_rows,
            )

        # Ensure Requirement 7 Rare-Event Tugboat preset rows exist in vessel_trajectories
        tug_count_row = conn.execute(
            "SELECT COUNT(*) FROM vessel_trajectories WHERE vessel_type = 'HARBOR_TUG_RARE_MANEUVER'"
        ).fetchone()
        tug_count = int(tug_count_row[0]) if tug_count_row else 0
        if tug_count == 0:
            tug_rows: list[tuple[object, ...]] = []
            for condition in CONDITIONS:
                for mitigated in (False, True):
                    for window_id in range(1, 7):
                        w_idx = window_id - 1
                        gt_track, unmit_track, mit_track, cone_poly = build_tracks(
                            condition, window_id, mitigated, "HARBOR_TUG_RARE_MANEUVER"
                        )
                        # High-G turn (COG delta = 64.2 deg/min) causes 385m..445m divergence from Tier-1 linear anchor
                        divergence_m = round(385.0 + 12.0 * w_idx, 1)
                        # Fresh Tier-2 feed (0.6 min age), yet Trust Gate scores it low (0.29) due to false-positive spike flag
                        staleness_min = 0.6
                        trust_score = 0.29

                        tug_rows.append(
                            (
                                condition,
                                mitigated,
                                window_id,
                                "367004890",
                                "TUG CAPE HENRY",
                                "HARBOR_TUG_RARE_MANEUVER",
                                json.dumps(gt_track),
                                json.dumps(unmit_track),
                                json.dumps(mit_track),
                                json.dumps(cone_poly),
                                divergence_m,
                                staleness_min,
                                trust_score,
                            )
                        )

            conn.executemany(
                """
                INSERT INTO vessel_trajectories (
                    condition,
                    mitigated,
                    window_id,
                    mmsi,
                    vessel_name,
                    vessel_type,
                    ground_truth_json,
                    unmitigated_pred_json,
                    mitigated_pred_json,
                    uncertainty_poly_json,
                    tier1_divergence_m,
                    staleness_min,
                    trust_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tug_rows,
            )

        # Seed baseline Watch Officer audit log entry if empty so avg review effort is immediately populated
        audit_count_row = conn.execute("SELECT COUNT(*) FROM hitl_audit_log").fetchone()
        audit_count = int(audit_count_row[0]) if audit_count_row else 0
        if audit_count == 0 and not real:  # no made-up review time next to real results
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            conn.execute(
                """
                INSERT INTO hitl_audit_log (
                    id,
                    timestamp,
                    mmsi,
                    vessel_name,
                    condition,
                    window_id,
                    action_taken,
                    review_duration_seconds,
                    operator_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    1,
                    now_utc,
                    "367591240",
                    "MV VIRGINIA TRADER",
                    "COMBINED",
                    5,
                    "DEFER_TO_ANCHOR_KINEMATIC",
                    8.4,
                    "Initial Watch Officer calibration verification against USCG Tier-1 coastal anchor.",
                ),
            )
    finally:
        conn.close()
