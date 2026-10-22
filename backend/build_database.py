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

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "backend" / "maritime_case.duckdb"
MANIFEST_PATH = ROOT_DIR / "backend" / "manifest.json"
MOD_LOG_PATH = ROOT_DIR / "backend" / "modification_log.json"
TEMP_DIR = ROOT_DIR / "data_temp"

# Bounding box & Filters for Chesapeake / Norfolk
LAT_MIN, LAT_MAX = 36.8, 37.3
LON_MIN, LON_MAX = -76.6, -75.8
MIN_SOG = 0.5

# NOAA Azure Storage base URL for 2025 daily AIS data
NOAA_BASE_URL = "https://noaaocm.blob.core.windows.net/ais/csv2/csv2025"


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
        voyage_id VARCHAR,
        mmsi BIGINT,
        timestamp TIMESTAMP,
        lat DOUBLE,
        lon DOUBLE,
        sog DOUBLE,
        cog DOUBLE,
        source_tier VARCHAR,    -- 'TIER1_ANCHOR' | 'TIER2_FAST'
        condition VARCHAR,      -- 'NOMINAL' | 'STALE' | 'LOW_QUALITY' | 'COMBINED'
        window_id INTEGER,      -- 1 to 6
        is_defective BOOLEAN    -- Hidden from model training/inference
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
    con: duckdb.DuckDBPyConnection, start_date: date, days: int = 42
):
  """Downloads daily slices for 6 weeks, decompresses, filters, and loads raw_reference_ais."""
  TEMP_DIR.mkdir(parents=True, exist_ok=True)

  # Stage filtered pings into a temporary table before voyage assignment
  con.execute("""
        CREATE TEMP TABLE temp_filtered_pings (
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
    date_str = curr_date.strftime("%Y_%m_%d")

    # Try .csv.zst first, fallback to .csv
    extensions = [".csv.zst", ".csv"]
    downloaded_path = None
    target_csv_path = None

    for ext in extensions:
      filename = f"AIS_{date_str}{ext}"
      url = f"{NOAA_BASE_URL}/{filename}"
      dest_file = TEMP_DIR / filename
      print(f"[{day_offset+1}/{days}] Fetching {filename}...")

      try:
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code == 200:
          with open(dest_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
              if chunk:
                f.write(chunk)
          downloaded_path = dest_file
          break
      except Exception as e:
        print(f"  Failed attempt from {url}: {e}")

    if not downloaded_path:
      print(f"  Warning: Could not fetch slice for {date_str}, skipping.")
      continue

    # Decompress if .csv.zst
    if str(downloaded_path).endswith(".zst"):
      target_csv_path = TEMP_DIR / f"AIS_{date_str}.csv"
      dctx = zstd.ZstdDecompressor()
      with (
          open(downloaded_path, "rb") as ifh,
          open(target_csv_path, "wb") as ofh,
      ):
        dctx.copy_stream(ifh, ofh)
      downloaded_path.unlink()  # remove .zst immediately
    else:
      target_csv_path = downloaded_path

    # Stream into DuckDB with bounding box filter applied during read
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

    # Clean up the uncompressed CSV from disk
    if target_csv_path.exists():
      target_csv_path.unlink()

  # Segment into leak-free voyages and assign 6 windows
  print("Segmenting voyages and assigning windows W1-W6...")
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

  if TEMP_DIR.exists():
    for f in TEMP_DIR.glob("*"):
      f.unlink()
    TEMP_DIR.rmdir()


def inject_degradations(con: duckdb.DuckDBPyConnection):
  """Splits streams into TIER1_ANCHOR and TIER2_FAST, injecting the 4 experimental conditions."""
  print(
      "Generating injected streams (NOMINAL, STALE, LOW_QUALITY, COMBINED)..."
  )

  df = con.execute("""
        SELECT voyage_id, mmsi, base_datetime, lat, lon, sog, cog, window_id
        FROM raw_reference_ais
        ORDER BY voyage_id, base_datetime
    """).df()

  if len(df) == 0:
    print("Warning: No records found to inject.")
    return

  np.random.seed(42)

  # 20% TIER1_ANCHOR voyages, 80% TIER2_FAST voyages
  unique_voyages = df["voyage_id"].unique()
  anchor_voyages = set(
      np.random.choice(
          unique_voyages, size=int(len(unique_voyages) * 0.2), replace=False
      )
  )
  df["source_tier"] = df["voyage_id"].apply(
      lambda v: "TIER1_ANCHOR" if v in anchor_voyages else "TIER2_FAST"
  )

  stale_rates = {1: 0.0, 2: 0.15, 3: 0.30, 4: 0.50, 5: 0.70, 6: 0.70}
  noise_sigmas = {
      1: 0.0,
      2: 0.0005,
      3: 0.0010,
      4: 0.0020,
      5: 0.0035,
      6: 0.0035,
  }

  records_to_insert = []
  mod_stats = {"total_rows": len(df), "conditions": {}}

  for cond in ["NOMINAL", "STALE", "LOW_QUALITY", "COMBINED"]:
    mod_stats["conditions"][cond] = {"stale_injected": 0, "noise_injected": 0}
    df_cond = df.copy()
    df_cond["condition"] = cond
    df_cond["is_defective"] = False

    for w_id in range(1, 7):
      mask = (df_cond["window_id"] == w_id) & (
          df_cond["source_tier"] == "TIER2_FAST"
      )
      sub_indices = df_cond[mask].index
      n_rows = len(sub_indices)
      if n_rows == 0:
        continue

      if cond in ["STALE", "COMBINED"] and w_id > 1:
        stale_frac = stale_rates[w_id]
        n_stale = int(n_rows * stale_frac)
        stale_idx = np.random.choice(sub_indices, size=n_stale, replace=False)

        lag_steps = 10
        for idx in stale_idx:
          curr_voyage = df_cond.at[idx, "voyage_id"]
          earlier = df_cond[
              (df_cond["voyage_id"] == curr_voyage) & (df_cond.index < idx)
          ]
          if len(earlier) >= lag_steps:
            chosen = earlier.iloc[-lag_steps]
            df_cond.at[idx, "lat"] = chosen["lat"]
            df_cond.at[idx, "lon"] = chosen["lon"]
            df_cond.at[idx, "sog"] = chosen["sog"]
            df_cond.at[idx, "cog"] = chosen["cog"]
            df_cond.at[idx, "is_defective"] = True
            mod_stats["conditions"][cond]["stale_injected"] += 1

      if cond in ["LOW_QUALITY", "COMBINED"] and w_id > 1:
        sigma = noise_sigmas[w_id]
        df_cond.loc[sub_indices, "lat"] += np.random.normal(
            0, sigma, size=n_rows
        )
        df_cond.loc[sub_indices, "lon"] += np.random.normal(
            0, sigma, size=n_rows
        )

        spike_idx = np.random.choice(
            sub_indices, size=int(n_rows * 0.15), replace=False
        )
        df_cond.loc[spike_idx, "sog"] += np.random.uniform(
            5.0, 15.0, size=len(spike_idx)
        )
        df_cond.loc[spike_idx, "cog"] = (
            df_cond.loc[spike_idx, "cog"]
            + np.random.uniform(30, 90, size=len(spike_idx))
        ) % 360
        df_cond.loc[sub_indices, "is_defective"] = True
        mod_stats["conditions"][cond]["noise_injected"] += len(sub_indices)

    records_to_insert.append(df_cond)

  import pandas as pd

  combined_df = pd.concat(records_to_insert, ignore_index=True)
  combined_df.rename(columns={"base_datetime": "timestamp"}, inplace=True)

  con.register(
      "tmp_injected",
      combined_df[[
          "voyage_id",
          "mmsi",
          "timestamp",
          "lat",
          "lon",
          "sog",
          "cog",
          "source_tier",
          "condition",
          "window_id",
          "is_defective",
      ]],
  )
  con.execute("INSERT INTO injected_streams SELECT * FROM tmp_injected")

  with open(MOD_LOG_PATH, "w") as f:
    json.dump(mod_stats, f, indent=2)


def export_manifest(con: duckdb.DuckDBPyConnection):
  """Exports manifest.json required by Requirement 2."""
  meta = {
      "dataset_name": (
          "MarineCadastre AIS 2025 (Chesapeake Bay & Norfolk Corridor)"
      ),
      "source_url": NOAA_BASE_URL,
      "retrieval_date": datetime.now().strftime("%Y-%m-%d"),
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
      "chronological_windows": 6,
      "total_reference_pings": con.execute(
          "SELECT COUNT(*) FROM raw_reference_ais"
      ).fetchone()[0],
      "unique_voyages": con.execute(
          "SELECT COUNT(DISTINCT voyage_id) FROM raw_reference_ais"
      ).fetchone()[0],
      "license": "Public Domain (U.S. Government NOAA/BOEM work)",
  }
  with open(MANIFEST_PATH, "w") as f:
    json.dump(meta, f, indent=2)


if __name__ == "__main__":
  con = duckdb.connect(str(DB_PATH))
  init_tables(con)

  # Target date range: 6 weeks starting Sep 1, 2025
  start_dt = date(2025, 9, 1)
  download_and_stream_ingest(con, start_date=start_dt, days=42)

  inject_degradations(con)
  export_manifest(con)
  con.close()
  print("Database pipeline complete.")