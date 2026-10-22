"""
backend/build_database.py - Automated Ingestion & Degradation Pipeline
Downloads NOAA 2025 daily AIS slices directly, filters for Chesapeake Bay / Norfolk,
populates maritime_case.duckdb, and generates injected streams and manifests.
"""

from datetime import date, timedelta
from datetime import datetime
import json
import os
from pathlib import Path
import duckdb
import numpy as np
import requests
import zstandard as zstd
import zipfile
import shutil
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_DIR / "backend" / "maritime_case.duckdb"
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
MOD_LOG_PATH = Path(__file__).resolve().parent / "modification_log.json"
TEMP_DIR = ROOT_DIR / "data_temp"

# Bounding box & Filters for Chesapeake / Norfolk
LAT_MIN, LAT_MAX = 36.8, 37.3
LON_MIN, LON_MAX = -76.6, -75.8
MIN_SOG = 0.5

# NOAA Azure Storage base URL for 2025 daily AIS data
NOAA_BASE_URL = "https://noaaocm.blob.core.windows.net/ais/csv2/csv2025"

#NOAA AIS files are large, so we set a custom User-Agent to avoid 403 errors
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Degradation settings (recorded in modification_log.json)
RANDOM_SEED = 42
TIER1_EVERY_N = 5  # every 5th ping of a voyage goes to the slow, accurate TIER1_ANCHOR feed (20%)
STALE_LAG_PINGS = 10  # a stale ping repeats the report from 10 pings earlier (~12 min)
STALE_RATES = {1: 0.0, 2: 0.15, 3: 0.30, 4: 0.50, 5: 0.70, 6: 0.70}
NOISE_SIGMA_DEG = {1: 0.0, 2: 0.0005, 3: 0.0010, 4: 0.0020, 5: 0.0035, 6: 0.0035}
SPIKE_FRACTION = 0.15  # share of noisy pings that also get SOG/COG spikes


def init_tables(con: duckdb.DuckDBPyConnection):
  """Initializes DuckDB tables per Architecture.md."""
  con.execute("DROP TABLE IF EXISTS raw_reference_ais;")
  con.execute("DROP TABLE IF EXISTS injected_streams;")
  con.execute("DROP TABLE IF EXISTS experiment_metrics;")
  con.execute("DROP TABLE IF EXISTS hitl_audit_log;")

  con.execute("""
    CREATE TABLE raw_reference_ais (
        voyage_id VARCHAR,
        mmsi BIGINT,
        base_datetime TIMESTAMP,
        lat DOUBLE,
        lon DOUBLE,
        sog DOUBLE,
        cog DOUBLE,
        heading DOUBLE,
        vessel_name VARCHAR,
        vessel_type VARCHAR,
        window_id INTEGER
    );

    CREATE TABLE injected_streams (
        mmsi BIGINT,
        voyage_id VARCHAR,
        timestamp TIMESTAMP,
        lat DOUBLE,
        lon DOUBLE,
        sog DOUBLE,
        cog DOUBLE,
        vessel_type VARCHAR,         -- ADDED: So Person 2 can filter ship types
        window_id INTEGER,           -- 1 to 6
        source_tier VARCHAR,         -- 'TIER1_ANCHOR' | 'TIER2_FAST'
        condition VARCHAR,           -- 'NOMINAL' | 'STALE' | 'LOW_QUALITY' | 'COMBINED'
        is_injected_defect BOOLEAN   -- RENAMED: Hidden from model training/inference
    );

    CREATE TABLE experiment_metrics (
        condition VARCHAR,
        mitigated BOOLEAN,
        "window" INTEGER,       -- Double-quoted to prevent syntax error
        displacement_mae_m DOUBLE,
        tail_95_error_m DOUBLE,
        data_discarded_pct DOUBLE,
        decisions_deferred_pct DOUBLE,
        model_compute_latency_ms DOUBLE
    );

    CREATE TABLE hitl_audit_log (
        audit_id VARCHAR,
        mmsi BIGINT,
        action VARCHAR,
        review_duration_seconds DOUBLE,
        condition VARCHAR,
        "window" INTEGER,
        logged_at_utc TIMESTAMP,
        model_compute_latency_ms DOUBLE
    );
    """)


def download_and_stream_ingest(
    con: duckdb.DuckDBPyConnection, start_date: date, days: int = 12
):
  """Downloads ais-YYYY-MM-DD.csv.zst files directly, filters for Chesapeake Bay, and loads raw_reference_ais."""
  TEMP_DIR.mkdir(parents=True, exist_ok=True)
  LOCAL_DATA_DIR = ROOT_DIR / "data_raw"

  con.execute("""
        CREATE TEMP TABLE IF NOT EXISTS temp_filtered_pings (
            mmsi BIGINT,
            base_datetime TIMESTAMP,
            lat DOUBLE,
            lon DOUBLE,
            sog DOUBLE,
            cog DOUBLE,
            heading DOUBLE,
            vessel_name VARCHAR,
            vessel_type VARCHAR
        );
    """)

  for day_offset in range(days):
    curr_date = start_date + timedelta(days=day_offset)
    date_str = curr_date.strftime("%Y-%m-%d")

    zst_filename = f"ais-{date_str}.csv.zst"
    csv_filename = f"ais-{date_str}.csv"

    dest_zst = TEMP_DIR / zst_filename
    target_csv_path = None

    # 1. Use local CSV if you already downloaded it
    local_csv_candidates = [
        LOCAL_DATA_DIR / csv_filename,
        ROOT_DIR / "data" / csv_filename,
    ]
    for cand in local_csv_candidates:
      if cand.exists():
        print(f"[{day_offset+1}/{days}] Using local file: {cand.name}")
        target_csv_path = cand
        break

    # 2. Otherwise download the .csv.zst file directly
    if not target_csv_path:
      url = f"{NOAA_BASE_URL}/{zst_filename}"
      print(f"[{day_offset+1}/{days}] Downloading {zst_filename}...")

      try:
        resp = requests.get(url, headers=HEADERS, stream=True, timeout=180)
        if resp.status_code == 200:
          with open(dest_zst, "wb") as f:
            for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
              if chunk:
                f.write(chunk)

          # Decompress .zst to .csv in TEMP_DIR
          target_csv_path = TEMP_DIR / csv_filename
          dctx = zstd.ZstdDecompressor()
          with (
              open(dest_zst, "rb") as ifh,
              open(target_csv_path, "wb") as ofh,
          ):
            dctx.copy_stream(ifh, ofh)

          dest_zst.unlink(missing_ok=True)  # remove compressed file immediately
        else:
          print(
              f"  Warning: HTTP {resp.status_code} for {zst_filename}, skipping."
          )
          continue
      except Exception as e:
        print(f"  Network error for {zst_filename}: {e}")
        continue

    # 3. Stream into DuckDB applying spatial and vessel filters
    if target_csv_path and target_csv_path.exists():
      print(f"  Filtering and inserting {target_csv_path.name} into DuckDB...")
      con.execute(f"""
                INSERT INTO temp_filtered_pings
                SELECT 
                    mmsi::BIGINT,
                    base_date_time::TIMESTAMP,
                    latitude::DOUBLE,
                    longitude::DOUBLE,
                    sog::DOUBLE,
                    cog::DOUBLE,
                    heading::DOUBLE,
                    vessel_name::VARCHAR,
                    vessel_type::VARCHAR
                FROM read_csv_auto('{target_csv_path.as_posix()}')
                WHERE latitude BETWEEN {LAT_MIN} AND {LAT_MAX}
                  AND longitude BETWEEN {LON_MIN} AND {LON_MAX}
                  AND sog > {MIN_SOG}
                  AND (
                      TRY_CAST(vessel_type AS INTEGER) IN (31, 32, 52) 
                      OR TRY_CAST(vessel_type AS INTEGER) BETWEEN 70 AND 89
                  );
            """)

      # Delete decompressed CSV if it was created in temp
      if target_csv_path.parent == TEMP_DIR:
        target_csv_path.unlink(missing_ok=True)

  # 4. Check if records were captured
  total_filtered = con.execute(
      "SELECT COUNT(*) FROM temp_filtered_pings"
  ).fetchone()[0]
  if total_filtered == 0:
    raise RuntimeError(
        "No records ingested. Check if the dates exist on the NOAA server or"
        " verify your bounding box."
    )

  # 5. Segment into leak-free voyages and partition across W1-W6
  print(
      f"Segmenting {total_filtered:,} pings into leak-free voyages across"
      " W1-W6..."
  )
  con.execute("""
        INSERT INTO raw_reference_ais
        WITH ordered_pings AS (
            SELECT *,
                LAG(base_datetime) OVER (PARTITION BY mmsi ORDER BY base_datetime) AS prev_time
            FROM temp_filtered_pings
        ),
        voyage_segmented AS (
            SELECT *,
                SUM(CASE WHEN prev_time IS NULL OR epoch(base_datetime) - epoch(prev_time) > 1800 THEN 1 ELSE 0 END)
                OVER (PARTITION BY mmsi ORDER BY base_datetime) AS voyage_num
            FROM ordered_pings
        ),
        voyages_assigned AS (
            SELECT *,
                mmsi || '_' || voyage_num AS voyage_id,
                MIN(base_datetime) OVER (PARTITION BY mmsi, voyage_num) AS voyage_start
            FROM voyage_segmented
        )
        SELECT 
            voyage_id, mmsi, base_datetime, lat, lon, sog, cog, heading, vessel_name, vessel_type,
            NTILE(6) OVER (ORDER BY voyage_start) AS window_id
        FROM voyages_assigned;
    """)

  # Clean up temp folder safely
  if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


def assign_source_tiers(df: pd.DataFrame) -> pd.Series:
  """Split each voyage's pings between the two feeds.

  Both feeds observe the SAME voyages, so the slow anchor feed can audit the
  fast feed (the countermeasure needs this overlap). Within each voyage, every
  TIER1_EVERY_N-th ping (1st, 6th, 11th, ...) is TIER1_ANCHOR (~20%, sparse,
  clean); all other pings are TIER2_FAST (~80%, the feed that degrades).
  `df` must be sorted by voyage_id, then time.
  """
  ping_number = df.groupby("voyage_id", sort=False).cumcount()
  return pd.Series(
      np.where(ping_number % TIER1_EVERY_N == 0, "TIER1_ANCHOR", "TIER2_FAST"),
      index=df.index,
  )


def inject_degradations(con: duckdb.DuckDBPyConnection):
  """Builds injected_streams: 4 conditions x 2 feeds, defects only on TIER2_FAST in W2-W6."""
  print("Generating injected streams (NOMINAL, STALE, LOW_QUALITY, COMBINED)...")

  df = con.execute("""
        SELECT voyage_id, mmsi, base_datetime, lat, lon, sog, cog, vessel_type, window_id
        FROM raw_reference_ais
        ORDER BY voyage_id, base_datetime
    """).df()

  if len(df) == 0:
    print("Warning: No records found to inject.")
    return

  rng = np.random.default_rng(RANDOM_SEED)
  df["source_tier"] = assign_source_tiers(df)

  # The report from STALE_LAG_PINGS pings earlier in the same voyage (NaN if none).
  earlier = df.groupby("voyage_id", sort=False)[["lat", "lon", "sog", "cog"]].shift(STALE_LAG_PINGS)
  has_earlier = earlier["lat"].notna().to_numpy()

  records_to_insert = []
  mod_stats = {
      "total_rows": len(df),
      "settings": {
          "random_seed": RANDOM_SEED,
          "tier_rule": f"every {TIER1_EVERY_N}th ping of each voyage -> TIER1_ANCHOR; rest -> TIER2_FAST",
          "defects_applied_to": "TIER2_FAST only, windows W2-W6; TIER1_ANCHOR and W1 stay clean",
          "stale_rule": f"lat/lon/sog/cog replaced by the report {STALE_LAG_PINGS} pings earlier, timestamp kept",
          "stale_rates_by_window": STALE_RATES,
          "noise_rule": "Gaussian noise on lat/lon (degrees) for every TIER2_FAST ping; "
                        f"{SPIKE_FRACTION:.0%} also get SOG +5-15 kn and COG +30-90 deg",
          "noise_sigma_deg_by_window": NOISE_SIGMA_DEG,
      },
      "conditions": {},
  }

  for cond in ["NOMINAL", "STALE", "LOW_QUALITY", "COMBINED"]:
    stats = {"stale_injected": 0, "noise_injected": 0, "by_window": {}}
    df_cond = df.copy()
    df_cond["condition"] = cond
    df_cond["is_injected_defect"] = False

    for w_id in range(1, 7):
      w_stats = {"tier2_rows": 0, "stale_injected": 0, "noise_injected": 0}
      tier2 = ((df_cond["window_id"] == w_id) & (df_cond["source_tier"] == "TIER2_FAST")).to_numpy()
      sub_indices = np.flatnonzero(tier2)
      n_rows = len(sub_indices)
      w_stats["tier2_rows"] = n_rows

      if n_rows > 0 and cond in ["STALE", "COMBINED"] and w_id > 1:
        n_stale = int(n_rows * STALE_RATES[w_id])
        picked = rng.choice(sub_indices, size=n_stale, replace=False)
        picked = picked[has_earlier[picked]]  # need 10 earlier pings in the voyage
        for col in ["lat", "lon", "sog", "cog"]:
          df_cond.iloc[picked, df_cond.columns.get_loc(col)] = earlier[col].to_numpy()[picked]
        df_cond.iloc[picked, df_cond.columns.get_loc("is_injected_defect")] = True
        w_stats["stale_injected"] = len(picked)

      if n_rows > 0 and cond in ["LOW_QUALITY", "COMBINED"] and w_id > 1:
        sigma = NOISE_SIGMA_DEG[w_id]
        lat_col, lon_col = df_cond.columns.get_loc("lat"), df_cond.columns.get_loc("lon")
        df_cond.iloc[sub_indices, lat_col] += rng.normal(0, sigma, size=n_rows)
        df_cond.iloc[sub_indices, lon_col] += rng.normal(0, sigma, size=n_rows)

        spikes = rng.choice(sub_indices, size=int(n_rows * SPIKE_FRACTION), replace=False)
        sog_col, cog_col = df_cond.columns.get_loc("sog"), df_cond.columns.get_loc("cog")
        df_cond.iloc[spikes, sog_col] += rng.uniform(5.0, 15.0, size=len(spikes))
        df_cond.iloc[spikes, cog_col] = (
            df_cond.iloc[spikes, cog_col].to_numpy() + rng.uniform(30, 90, size=len(spikes))
        ) % 360
        df_cond.iloc[sub_indices, df_cond.columns.get_loc("is_injected_defect")] = True
        w_stats["noise_injected"] = n_rows

      stats["stale_injected"] += w_stats["stale_injected"]
      stats["noise_injected"] += w_stats["noise_injected"]
      stats["by_window"][w_id] = w_stats

    mod_stats["conditions"][cond] = stats
    records_to_insert.append(df_cond)

  combined_df = pd.concat(records_to_insert, ignore_index=True)
  combined_df = combined_df.rename(columns={"base_datetime": "timestamp"})

  con.execute("DELETE FROM injected_streams")  # rebuild from scratch, never append twice
  con.register(
      "tmp_injected",
      combined_df[[
          "mmsi",
          "voyage_id",
          "timestamp",
          "lat",
          "lon",
          "sog",
          "cog",
          "vessel_type",
          "window_id",
          "source_tier",
          "condition",
          "is_injected_defect",
      ]],
  )
  con.execute("INSERT INTO injected_streams SELECT * FROM tmp_injected")
  con.unregister("tmp_injected")

  # Problems already present in the source data (not introduced by us), logged separately.
  mod_stats["observed_source_issues"] = {
      "duplicate_timestamps_within_voyage": int(con.execute("""
          SELECT COUNT(*) FROM (SELECT voyage_id, base_datetime FROM raw_reference_ais
                                GROUP BY ALL HAVING COUNT(*) > 1)""").fetchone()[0]),
      "rows_with_missing_cog": int(con.execute(
          "SELECT COUNT(*) FROM raw_reference_ais WHERE cog IS NULL").fetchone()[0]),
      "voyages_spanning_two_windows": int(con.execute("""
          SELECT COUNT(*) FROM (SELECT voyage_id FROM raw_reference_ais
                                GROUP BY 1 HAVING COUNT(DISTINCT window_id) > 1)""").fetchone()[0]),
  }

  with open(MOD_LOG_PATH, "w") as f:
    json.dump(mod_stats, f, indent=2)


def export_manifest(con: duckdb.DuckDBPyConnection, retrieval_date: str | None = None):
  """Exports manifest.json (Requirement 2 / minimum data manifest), computed from the database."""
  one = lambda sql: con.execute(sql).fetchone()[0]
  first_ts, last_ts = con.execute(
      "SELECT MIN(base_datetime), MAX(base_datetime) FROM raw_reference_ais").fetchone()
  windows = con.execute("""
      SELECT window_id, COUNT(DISTINCT voyage_id), COUNT(*), MIN(base_datetime), MAX(base_datetime)
      FROM raw_reference_ais GROUP BY 1 ORDER BY 1""").fetchall()
  vessel_types = con.execute("""
      SELECT vessel_type, COUNT(DISTINCT voyage_id) FROM raw_reference_ais
      GROUP BY 1 ORDER BY 2 DESC""").fetchall()
  meta = {
      "dataset_name": (
          "MarineCadastre AIS 2025 (Chesapeake Bay & Norfolk Corridor)"
      ),
      "source_url": NOAA_BASE_URL,
      "retrieval_date": retrieval_date or datetime.now().strftime("%Y-%m-%d"),
      "date_range_utc": [str(first_ts), str(last_ts)],
      "spatial_bounding_box": {
          "lat_min": LAT_MIN,
          "lat_max": LAT_MAX,
          "lon_min": LON_MIN,
          "lon_max": LON_MAX,
      },
      "vessel_cohorts": [
          "Tug/Towing (31, 32, 52)",
          "Cargo (70-79)",
          "Tanker (80-89)",
      ],
      "minimum_sog_knots": MIN_SOG,
      "voyage_rule": "new voyage when the same MMSI has a gap > 30 min between pings",
      "window_rule": "NTILE(6) over pings ordered by voyage start time (equal ping counts per window)",
      "chronological_windows": 6,
      "total_reference_pings": one("SELECT COUNT(*) FROM raw_reference_ais"),
      "unique_voyages": one("SELECT COUNT(DISTINCT voyage_id) FROM raw_reference_ais"),
      "unique_vessels": one("SELECT COUNT(DISTINCT mmsi) FROM raw_reference_ais"),
      "windows": [
          {"window_id": w, "voyages": v, "pings": n, "first_ping_utc": str(a), "last_ping_utc": str(b)}
          for w, v, n, a, b in windows
      ],
      "voyages_by_vessel_type_code": {str(t): n for t, n in vessel_types},
      "missing_value_treatment": "pings with SOG <= 0.5 kn removed; rows with missing COG kept as-is",
      "injected_streams": "see modification_log.json (simulated degradations, not real feed failures)",
      "license": "Public Domain (U.S. Government NOAA/BOEM work)",
  }
  with open(MANIFEST_PATH, "w") as f:
    json.dump(meta, f, indent=2)


if __name__ == "__main__":
  # Full build:   uv run python backend/database/build_database.py
  # Re-inject only (keeps downloaded data, rebuilds injected_streams + manifest + log):
  #               uv run python backend/database/build_database.py --reinject
  con = duckdb.connect(str(DB_PATH))
  if "--reinject" in sys.argv:
    previous_date = None
    if MANIFEST_PATH.exists():
      previous_date = json.loads(MANIFEST_PATH.read_text()).get("retrieval_date")
    inject_degradations(con)
    export_manifest(con, retrieval_date=previous_date)
  else:
    init_tables(con)

    # Target date range: 12 days starting Sep 1, 2025
    start_dt = date(2025, 9, 1)
    download_and_stream_ingest(con, start_date=start_dt, days=12)

    inject_degradations(con)
    export_manifest(con)
  con.close()
  print("Database pipeline complete.")
