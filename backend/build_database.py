"""
backend/build_database.py
Combines Step 3 (Ingest & Partition) and Step 4 (Degradation Injector)
into maritime_case.duckdb with zero data leakage.
"""
import duckdb
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = ROOT_DIR / "backend" / "maritime_case.duckdb"
MANIFEST_PATH = ROOT_DIR / "backend" / "manifest.json"
MOD_LOG_PATH = ROOT_DIR / "backend" / "modification_log.json"

# Bounding box & Filters for Chesapeake / Norfolk
LAT_MIN, LAT_MAX = 36.8, 37.3
LON_MIN, LON_MAX = -76.6, -75.8
MIN_SOG = 0.5  # Knots (filter out moored/stationary boats)

def init_tables(con: duckdb.DuckDBPyConnection):
    """Initializes DuckDB schema as required by Architecture.md."""
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
        "window" INTEGER,
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

def ingest_and_segment_voyages(con: duckdb.DuckDBPyConnection):
    """
    Step 3: Streams raw CSVs directly from disk, filters to Norfolk/Chesapeake,
    segments pings into distinct voyages (30+ min gaps), and maps them into 6 windows.
    """
    print("Ingesting raw AIS data into DuckDB...")
    
    # Matches ais-2025-09-01.csv, or any files in data/
    csv_pattern = str(DATA_DIR / "*.csv")
    
    query = f"""
    INSERT INTO raw_reference_ais
    WITH filtered_pings AS (
        SELECT 
            mmsi::BIGINT AS mmsi,
            base_date_time::TIMESTAMP AS base_datetime,
            latitude::DOUBLE AS lat,
            longitude::DOUBLE AS lon,
            sog::DOUBLE AS sog,
            cog::DOUBLE AS cog,
            heading::DOUBLE AS heading,
            vessel_name::VARCHAR AS vessel_name,
            vessel_type::VARCHAR AS vessel_type,
            LAG(base_date_time::TIMESTAMP) OVER (
                PARTITION BY mmsi ORDER BY base_date_time::TIMESTAMP
            ) AS prev_time
        FROM read_csv_auto('{csv_pattern}')
        WHERE latitude BETWEEN {LAT_MIN} AND {LAT_MAX}
          AND longitude BETWEEN {LON_MIN} AND {LON_MAX}
          AND sog > {MIN_SOG}
          AND (
              TRY_CAST(vessel_type AS INTEGER) IN (31, 32, 52) 
              OR TRY_CAST(vessel_type AS INTEGER) BETWEEN 70 AND 89
          )
    ),
    voyage_segmented AS (
        SELECT *,
            SUM(CASE WHEN prev_time IS NULL OR epoch(base_datetime) - epoch(prev_time) > 1800 THEN 1 ELSE 0 END)
            OVER (PARTITION BY mmsi ORDER BY base_datetime) AS voyage_num
        FROM filtered_pings
    ),
    voyages_assigned AS (
        SELECT *,
            mmsi || '_' || voyage_num AS voyage_id,
            MIN(base_datetime) OVER (PARTITION BY mmsi, voyage_num) AS voyage_start
        FROM voyage_segmented
    )
    SELECT 
        voyage_id, mmsi, base_datetime, lat, lon, sog, cog, heading, vessel_name, vessel_type,
        -- Ensure entire voyages stay in one window by ordering on voyage_start
        NTILE(6) OVER (ORDER BY voyage_start) AS window_id
    FROM voyages_assigned;
    """
    con.execute(query)
    count = con.execute("SELECT COUNT(*) FROM raw_reference_ais").fetchone()[0]
    voyages = con.execute("SELECT COUNT(DISTINCT voyage_id) FROM raw_reference_ais").fetchone()[0]
    print(f"Extracted {count:,} pings across {voyages:,} unique voyages for Chesapeake Bay.")

def inject_degradations(con: duckdb.DuckDBPyConnection):
    """
    Step 4: Implements the 4 experimental conditions and splits into
    TIER1_ANCHOR (20%) and TIER2_FAST (80%) across windows W1-W6.
    """
    print("Generating injected streams (Nominal, Stale, Low-Quality, Combined)...")
    
    # Pull raw rows ordered by voyage and time
    df = con.execute("""
        SELECT voyage_id, mmsi, base_datetime, lat, lon, sog, cog, window_id
        FROM raw_reference_ais
        ORDER BY voyage_id, base_datetime
    """).df()

    if len(df) == 0:
        print("Warning: No records found to inject.")
        return

    np.random.seed(42)  # Fixed seed for reproducibility
    
    # 20% TIER1_ANCHOR, 80% TIER2_FAST assigned per voyage
    unique_voyages = df['voyage_id'].unique()
    anchor_voyages = set(np.random.choice(unique_voyages, size=int(len(unique_voyages) * 0.2), replace=False))
    df['source_tier'] = df['voyage_id'].apply(lambda v: 'TIER1_ANCHOR' if v in anchor_voyages else 'TIER2_FAST')

    # Degradation progression schedules across windows 1 through 6
    stale_rates = {1: 0.0, 2: 0.15, 3: 0.30, 4: 0.50, 5: 0.70, 6: 0.70}
    noise_sigmas = {1: 0.0, 2: 0.0005, 3: 0.0010, 4: 0.0020, 5: 0.0035, 6: 0.0035}  # ~50m to 350m jitter

    records_to_insert = []
    mod_stats = {"total_rows": len(df), "conditions": {}}

    for cond in ['NOMINAL', 'STALE', 'LOW_QUALITY', 'COMBINED']:
        mod_stats["conditions"][cond] = {"stale_injected": 0, "noise_injected": 0}
        df_cond = df.copy()
        df_cond['condition'] = cond
        df_cond['is_defective'] = False

        for w_id in range(1, 7):
            mask = (df_cond['window_id'] == w_id) & (df_cond['source_tier'] == 'TIER2_FAST')
            sub_indices = df_cond[mask].index
            n_rows = len(sub_indices)
            if n_rows == 0:
                continue

            # 1. Stale injection (replaying coords from earlier in the voyage)
            if cond in ['STALE', 'COMBINED'] and w_id > 1:
                stale_frac = stale_rates[w_id]
                n_stale = int(n_rows * stale_frac)
                stale_idx = np.random.choice(sub_indices, size=n_stale, replace=False)
                
                # Shift coordinates from 10 steps back
                lag_steps = 10
                for idx in stale_idx:
                    curr_voyage = df_cond.at[idx, 'voyage_id']
                    earlier_candidates = df_cond[(df_cond['voyage_id'] == curr_voyage) & (df_cond.index < idx)]
                    if len(earlier_candidates) >= lag_steps:
                        chosen = earlier_candidates.iloc[-lag_steps]
                        df_cond.at[idx, 'lat'] = chosen['lat']
                        df_cond.at[idx, 'lon'] = chosen['lon']
                        df_cond.at[idx, 'sog'] = chosen['sog']
                        df_cond.at[idx, 'cog'] = chosen['cog']
                        df_cond.at[idx, 'is_defective'] = True
                        mod_stats["conditions"][cond]["stale_injected"] += 1

            # 2. Low-Quality injection (GPS multipath noise + kinematic spikes)
            if cond in ['LOW_QUALITY', 'COMBINED'] and w_id > 1:
                sigma = noise_sigmas[w_id]
                noise_lat = np.random.normal(0, sigma, size=n_rows)
                noise_lon = np.random.normal(0, sigma, size=n_rows)
                
                df_cond.loc[sub_indices, 'lat'] += noise_lat
                df_cond.loc[sub_indices, 'lon'] += noise_lon
                
                # SOG/COG sensor spikes on 15% of records
                spike_idx = np.random.choice(sub_indices, size=int(n_rows * 0.15), replace=False)
                df_cond.loc[spike_idx, 'sog'] += np.random.uniform(5.0, 15.0, size=len(spike_idx))
                df_cond.loc[spike_idx, 'cog'] = (df_cond.loc[spike_idx, 'cog'] + np.random.uniform(30, 90, size=len(spike_idx))) % 360
                df_cond.loc[sub_indices, 'is_defective'] = True
                mod_stats["conditions"][cond]["noise_injected"] += len(sub_indices)

        records_to_insert.append(df_cond)

    full_injected_df = con.execute("SELECT * FROM df_cond LIMIT 0").df()
    import pandas as pd
    combined_df = pd.concat(records_to_insert, ignore_index=True)
    combined_df.rename(columns={'base_datetime': 'timestamp'}, inplace=True)
    
    con.register("tmp_injected", combined_df[[
        'voyage_id', 'mmsi', 'timestamp', 'lat', 'lon', 'sog', 'cog',
        'source_tier', 'condition', 'window_id', 'is_defective'
    ]])
    con.execute("INSERT INTO injected_streams SELECT * FROM tmp_injected")
    print("Injected streams successfully generated and loaded.")

    # Save modification log
    with open(MOD_LOG_PATH, "w") as f:
        json.dump(mod_stats, f, indent=2)
    print(f"Logged injection stats to {MOD_LOG_PATH}")

def export_manifest(con: duckdb.DuckDBPyConnection):
    """Generates the data manifest required for governance."""
    meta = {
        "dataset_name": "MarineCadastre AIS 2025 (Chesapeake Bay & Norfolk Corridor)",
        "source_url": "https://noaaocm.blob.core.windows.net/ais/csv2/csv2025/index.html",
        "retrieval_date": datetime.now().strftime("%Y-%m-%d"),
        "spatial_bounding_box": {
            "lat_min": LAT_MIN, "lat_max": LAT_MAX,
            "lon_min": LON_MIN, "lon_max": LON_MAX
        },
        "vessel_cohorts": ["Tug/Towing (31, 32, 52)", "Cargo (70-79)", "Tanker (80-89)"],
        "minimum_sog_knots": MIN_SOG,
        "chronological_windows": 6,
        "total_reference_pings": con.execute("SELECT COUNT(*) FROM raw_reference_ais").fetchone()[0],
        "unique_voyages": con.execute("SELECT COUNT(DISTINCT voyage_id) FROM raw_reference_ais").fetchone()[0],
        "license": "Public Domain (U.S. Government NOAA/BOEM work)"
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported data governance manifest to {MANIFEST_PATH}")

if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    init_tables(con)
    ingest_and_segment_voyages(con)
    inject_degradations(con)
    export_manifest(con)
    con.close()
    print("All tasks for Person 1 completed successfully!")