"""Load AIS pings for the ML pipeline: Person 1's real database, or dummy data as a backup.

Every loader returns the agreed data contract, one row per ping:
    mmsi, voyage_id, timestamp (UTC), lat, lon, sog, cog, vessel_type,
    window_id, source_tier, condition
`is_injected_defect` is NOT returned: it says which rows we corrupted on purpose,
so the model must never see it. Use defect_share_by_window() for reporting.

Cleanup applied to real data (logged in the returned `.attrs["notes"]`):
  * A voyage belongs to the window of its FIRST ping. (A few voyages in the
    database have pings in two windows; this keeps each voyage whole.)
  * Rows missing lat, lon, sog or cog are dropped (the physics guess needs them).
  * Pings that share the exact same timestamp in the same voyage are dropped
    (both copies). The database has no rule for which copy is "first", so
    keeping one would make results change from run to run.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from backend.ml.dummy_data import generate_dummy_ais

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "backend" / "maritime_case.duckdb"  # Person 1's database

CONDITIONS = ["NOMINAL", "STALE", "LOW_QUALITY", "COMBINED"]
SOURCES = ["dummy", "real"]
MODEL_COLUMNS = [
    "mmsi", "voyage_id", "timestamp", "lat", "lon", "sog", "cog",
    "vessel_type", "window_id", "source_tier", "condition",
]


def connect(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open Person 1's database read-only (we never change their data from the ML side)."""
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Get maritime_case.duckdb from Person 1 "
            "or rebuild it with: uv run python backend/database/build_database.py"
        )
    return duckdb.connect(str(db_path), read_only=True)


# One SQL query does the window fix and the filtering inside DuckDB (fast, no big copies).
_REAL_QUERY = """
WITH stream AS (
    SELECT * FROM injected_streams WHERE condition = $condition
),
voyage_window AS (
    -- the window of each voyage's first ping
    SELECT voyage_id, arg_min(window_id, timestamp) AS first_window
    FROM stream GROUP BY voyage_id
)
SELECT
    CAST(s.mmsi AS VARCHAR) AS mmsi, s.voyage_id, s.timestamp, s.lat, s.lon, s.sog, s.cog,
    s.vessel_type, v.first_window AS window_id, s.source_tier, s.condition
FROM stream s JOIN voyage_window v USING (voyage_id)
WHERE list_contains($windows, v.first_window)
ORDER BY s.voyage_id, s.timestamp
"""


def load_pings(
    source: str = "real",
    condition: str = "NOMINAL",
    windows: list[int] | None = None,
    seed: int = 42,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Return pings for one condition and the chosen windows (default: all six)."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    windows = list(windows) if windows is not None else [1, 2, 3, 4, 5, 6]

    if source == "dummy":
        if condition != "NOMINAL":
            raise ValueError("Dummy data is clean only; use source='real' for poisoned conditions.")
        df = generate_dummy_ais(seed=seed)
        df = df[df["window_id"].isin(windows)].reset_index(drop=True)
        df.attrs["notes"] = {"source": "dummy (synthetic)"}
        return df[MODEL_COLUMNS]

    with connect(db_path) as con:
        df = con.execute(_REAL_QUERY, {"condition": condition, "windows": windows}).df()

    df = _clean_real(df)
    df.attrs["notes"] = {
        "source": f"real ({Path(db_path).name}, table injected_streams)",
        "condition": condition,
        "windows": windows,
        "rows_dropped_missing_values": df.attrs["dropped_missing"],
        "rows_dropped_duplicate_timestamps": df.attrs["dropped_duplicates"],
        "window_rule": "voyage assigned to the window of its first ping",
    }
    return df[MODEL_COLUMNS]


def _clean_real(df: pd.DataFrame) -> pd.DataFrame:
    """UTC timestamps; drop rows with missing values and all copies of duplicate timestamps."""
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")  # stored as UTC, no zone
    missing = df[["lat", "lon", "sog", "cog"]].isna().any(axis=1)
    duplicate = df.duplicated(["voyage_id", "timestamp"], keep=False)
    out = df[~missing & ~duplicate].reset_index(drop=True)
    out.attrs["dropped_missing"] = int(missing.sum())
    out.attrs["dropped_duplicates"] = int((duplicate & ~missing).sum())
    return out


def load_audit_labels(condition: str, windows: list[int] | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    """AUDIT ONLY: which rows we corrupted on purpose, plus the clean values behind them.

    Used to GRADE the trust gate after it has made its decisions. Never pass this to
    the gate or the model. Same row filtering as load_pings, so rows line up 1:1.
    Returns voyage_id, timestamp, is_injected_defect, and clean lat/lon/sog/cog.
    """
    windows = list(windows) if windows is not None else [1, 2, 3, 4, 5, 6]
    query = _REAL_QUERY.replace(
        "s.source_tier, s.condition",
        "s.source_tier, s.condition, s.is_injected_defect, "
        "r.lat AS clean_lat, r.lon AS clean_lon, r.sog AS clean_sog, r.cog AS clean_cog",
    ).replace(
        "FROM stream s JOIN voyage_window v USING (voyage_id)",
        "FROM stream s JOIN voyage_window v USING (voyage_id) "
        "JOIN raw_reference_ais r ON r.voyage_id = s.voyage_id AND r.base_datetime = s.timestamp",
    )
    with connect(db_path) as con:
        df = con.execute(query, {"condition": condition, "windows": windows}).df()
    df = _clean_real(df)
    return df[["voyage_id", "timestamp", "is_injected_defect", "clean_lat", "clean_lon", "clean_sog", "clean_cog"]]


def defect_share_by_window(condition: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """For REPORTING only: rows and % of rows we corrupted, per window (first-ping rule).

    This reads `is_injected_defect`, so its output must never be fed to the model.
    """
    query = """
    WITH stream AS (SELECT * FROM injected_streams WHERE condition = $condition),
    voyage_window AS (
        SELECT voyage_id, arg_min(window_id, timestamp) AS first_window FROM stream GROUP BY voyage_id
    )
    SELECT v.first_window AS window_id,
           COUNT(*) AS rows,
           SUM(CAST(s.is_injected_defect AS INTEGER)) AS corrupted_rows,
           SUM(CASE WHEN s.source_tier = 'TIER2_FAST' THEN 1 ELSE 0 END) AS tier2_rows
    FROM stream s JOIN voyage_window v USING (voyage_id)
    GROUP BY 1 ORDER BY 1
    """
    with connect(db_path) as con:
        df = con.execute(query, {"condition": condition}).df()
    df["corrupted_pct"] = 100.0 * df["corrupted_rows"] / df["rows"]
    df["tier2_corrupted_pct"] = 100.0 * df["corrupted_rows"] / df["tier2_rows"]
    return df
