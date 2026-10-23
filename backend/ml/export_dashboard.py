"""Step 8: export real results in the exact format Person 3's dashboard reads.

Run from the repository root, AFTER rolling.py:
    uv run python -m backend.ml.export_dashboard                                 # JSON file only
    uv run python -m backend.ml.export_dashboard --into data/maritime_case.duckdb  # also load the app's DB

Output: results/dashboard_export_real.json with two lists whose field names match the
tables in backend/db.py:

  experiment_metrics   one row per (condition, gate off/on, window W1-W6) = 48 rows
  vessel_trajectories  same keys: one example ship per window for the map

What each window means:
  W1      reference baseline: trained and tested inside W1, before any poisoning.
          Identical for all 8 runs (nothing has been corrupted yet).
  W2-W6   rolling.py: retrained on W1..W(k-1), tested on clean Wk.

Field notes (also written into the file):
  tier2_corrupted_volume_pct  % of fast-feed pings corrupted in the NEWEST training week
  avg_data_staleness_min      ESTIMATED average age of fast-feed reports in that week:
                              (share of stale reports) x (typical age of a stale report),
                              from backend/database/modification_log.json
  model_compute_latency_ms    median time for ONE prediction (one ship), measured here;
                              with the countermeasure it includes the 4 range models
  example ship                the cargo voyage with the most predictions in that test
                              window (a fixed rule, not picked for looking good)
  tracks                      ground truth = true position 15 min after each ping;
                              predictions = predicted position 15 min after each ping
  uncertainty polygon         a band around the gate-on prediction, as wide as each
                              prediction's 90% uncertainty radius; empty with the gate off
                              (no range models without the countermeasure)
  trust / divergence          what the gate sees in that ship's fast-feed pings for the
                              selected condition and window (median distance from the
                              anchor track, mean trust score)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from backend.ml.data import CONDITIONS, REPO_ROOT, connect, load_pings
from backend.ml.features import FEATURE_COLUMNS, make_dataset
from backend.ml.gate import compare_to_anchor, learn_limits, score_trust
from backend.ml.metrics import offset_m_to_latlon
from backend.ml.model_store import MODELS_DIR, load_models
from backend.ml.rolling import TEST_WINDOWS, model_name, training_buffer
from backend.ml.train_baseline import lgbm_params, predict_offsets, split_by_voyage_in_time
from backend.ml.uncertainty import SAFETY_RADIUS_M, predict_range, train_uncertainty_models, uncertainty_radius_m

RESULTS = REPO_ROOT / "results"
MOD_LOG = REPO_ROOT / "backend" / "database" / "modification_log.json"
STALE_LAG_PINGS = 10  # Person 1's stale rule (backend/database/build_database.py)
MAX_TRACK_POINTS = 40
SEED = 42


# --- helpers -------------------------------------------------------------------------
def pick_example_voyage(test: pd.DataFrame) -> str:
    """The cargo voyage with the most predictions (ties: smallest voyage_id). A fixed rule."""
    cargo = test[test["vessel_group"] == 0]
    pool = cargo if len(cargo) else test
    counts = pool.groupby("voyage_id").size().reset_index(name="n")
    return counts.sort_values(["n", "voyage_id"], ascending=[False, True]).iloc[0]["voyage_id"]


def thin(idx: np.ndarray, k: int = MAX_TRACK_POINTS) -> np.ndarray:
    """At most k evenly spaced positions, always keeping the first and last."""
    return idx if len(idx) <= k else idx[np.unique(np.linspace(0, len(idx) - 1, k).round().astype(int))]


def band_polygon(lat: np.ndarray, lon: np.ndarray, radius_m: np.ndarray) -> list[list[float]]:
    """A band around a track: each point pushed left and right by its radius, as one polygon."""
    n = len(lat)
    if n < 2:
        return []
    i_prev, i_next = np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, n - 1)
    d_north = (lat[i_next] - lat[i_prev]) * 111_195.0
    d_east = (lon[i_next] - lon[i_prev]) * 111_195.0 * np.cos(np.radians(lat))
    norm = np.hypot(d_north, d_east)
    norm[norm == 0] = 1.0
    left_n, left_e = -d_east / norm, d_north / norm  # unit vector to the left of travel
    l_lat, l_lon = offset_m_to_latlon(lat, lon, left_n * radius_m, left_e * radius_m)
    r_lat, r_lon = offset_m_to_latlon(lat, lon, -left_n * radius_m, -left_e * radius_m)
    pts = list(zip(l_lat, l_lon)) + list(zip(r_lat[::-1], r_lon[::-1]))
    return [[round(float(a), 6), round(float(b), 6)] for a, b in pts]


def track(lat, lon) -> list[list[float]]:
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lat, lon)]


def predicted_track(models: dict, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    north, east = predict_offsets(models, rows)
    return offset_m_to_latlon(rows["lat"], rows["lon"], north, east)


def single_prediction_ms(bundle: dict, row: pd.DataFrame, repeats: int = 300) -> float:
    """Median time to predict ONE ship (plus its range when the bundle has range models)."""
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        predict_offsets(bundle, row)
        if "uncertainty" in bundle:
            predict_range(bundle["uncertainty"], row)
        times.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(times)), 3)


def staleness_estimates() -> dict[tuple[str, int], tuple[float, float]]:
    """(condition, window) -> (% fast pings corrupted, estimated avg report age in minutes)."""
    log = json.loads(MOD_LOG.read_text())
    clean = load_pings("real", "NOMINAL")
    lag_s = clean.groupby("voyage_id")["timestamp"].diff(STALE_LAG_PINGS).dt.total_seconds()
    fast = clean["source_tier"] == "TIER2_FAST"
    typical_lag_min = (lag_s[fast].groupby(clean.loc[fast, "window_id"]).mean() / 60).to_dict()
    out = {}
    for cond in CONDITIONS:
        by_w = log["conditions"][cond]["by_window"]
        for w in range(1, 7):
            s = by_w[str(w)]
            rows = max(s["tier2_rows"], 1)
            corrupted = max(s["stale_injected"], s["noise_injected"])  # noise covers every corrupted row
            stale_share = s["stale_injected"] / rows
            out[(cond, w)] = (round(100.0 * corrupted / rows, 2), round(stale_share * typical_lag_min.get(w, 0.0), 2))
    return out


def vessel_names() -> dict[str, tuple[str, str]]:
    with connect() as con:
        rows = con.execute("""SELECT voyage_id, ANY_VALUE(mmsi), ANY_VALUE(vessel_name)
                              FROM raw_reference_ais GROUP BY 1""").fetchall()
    return {v: (str(m), (n or f"MMSI {m}").strip()) for v, m, n in rows}


def gate_view(cond: str, window: int, voyage: str, limits: dict) -> tuple[float, float]:
    """What the gate sees in this voyage's fast-feed pings: median anchor distance, mean trust."""
    pings = load_pings("real", cond, windows=[window])
    pings = pings[pings["voyage_id"] == voyage]
    scored = score_trust(compare_to_anchor(pings), limits)
    fast = scored[scored["verifiable"]]
    if fast.empty:
        return 0.0, 1.0
    return round(float(fast["anchor_dist_m"].median()), 1), round(float(fast["trust_score"].mean()), 3)


# --- build -----------------------------------------------------------------------------
def build() -> dict:
    rolling = json.loads((RESULTS / "rolling_real.json").read_text())
    rows = pd.DataFrame(rolling["rows"]).set_index(["condition", "mitigated", "window_id"])
    stale = staleness_estimates()
    names = vessel_names()

    # W1 reference: the baseline model (trained and tested inside W1), plus range models for its cone.
    w1 = make_dataset(load_pings("real", "NOMINAL", windows=[1]))
    w1_train, w1_test = split_by_voyage_in_time(w1)
    base = load_models(MODELS_DIR / "baseline_w1_real.joblib")["models"]
    base_range = train_uncertainty_models(w1_train, lgbm_params(SEED))
    baseline = json.loads((RESULTS / "baseline_w1_real.json").read_text())["lightgbm"]
    assert baseline["n_predictions"] == len(w1_test), "baseline file is out of date: rerun train_baseline --source real"

    tests = {1: w1_test, **{w: make_dataset(load_pings("real", "NOMINAL", windows=[w])) for w in TEST_WINDOWS}}
    examples = {w: pick_example_voyage(t) for w, t in tests.items()}

    # Gate limits for the "what the gate sees" fields (learned from W1, as in rolling.py).
    retrain_w2 = tests[2]["timestamp"].min()
    limits = {c: learn_limits(compare_to_anchor(training_buffer(load_pings("real", c, windows=[1]), 2, retrain_w2)))
              for c in CONDITIONS}

    metrics, trajectories = [], []
    latency = {}
    for cond in CONDITIONS:
        for w in range(1, 7):
            test, voyage = tests[w], examples[w]
            ex = test[test["voyage_id"] == voyage].sort_values("timestamp")
            ex = ex.iloc[thin(np.arange(len(ex)))]
            if w == 1:
                bundles = {False: base, True: {**base, "uncertainty": base_range}}
            else:
                bundles = {m: load_models(MODELS_DIR / f"{model_name('gated' if m else 'unmitigated', cond, 'real', w)}.joblib")["models"]
                           for m in (False, True)}
            off_lat, off_lon = predicted_track(bundles[False], ex)
            on_lat, on_lon = predicted_track(bundles[True], ex)
            radius = uncertainty_radius_m(predict_range(bundles[True]["uncertainty"], ex))
            divergence, trust = gate_view(cond, w, voyage, limits[cond])
            mmsi, vname = names.get(voyage, (voyage.split("_")[0], voyage))

            for mitigated in (False, True):
                if mitigated not in latency:
                    latency[mitigated] = single_prediction_ms(bundles[mitigated], ex.iloc[[0]])
                newest = 1 if w == 1 else w - 1  # newest week in the training data
                corrupted_pct, staleness = stale[(cond, newest)]
                if w == 1:
                    r = {"displacement_mae_m": baseline["displacement_mae_m"], "tail_95_error_m": baseline["tail_95_error_m"],
                         "data_discarded_pct": 0.0, "decisions_deferred_pct": 0.0}
                else:
                    r = rows.loc[(cond, mitigated, w)]
                metrics.append({
                    "condition": cond, "mitigated": mitigated, "window_id": w,
                    "displacement_mae_m": round(float(r["displacement_mae_m"]), 1),
                    "tail_95_error_m": round(float(r["tail_95_error_m"]), 1),
                    "data_discarded_pct": float(r["data_discarded_pct"]),
                    "decisions_deferred_pct": float(r["decisions_deferred_pct"]),
                    "tier2_corrupted_volume_pct": corrupted_pct,
                    "avg_data_staleness_min": staleness,
                    "model_compute_latency_ms": latency[mitigated],
                    "row_kind": "reference_baseline_W1" if w == 1 else "rolling_retrain",
                })
                trajectories.append({
                    "condition": cond, "mitigated": mitigated, "window_id": w,
                    "mmsi": mmsi, "vessel_name": vname, "vessel_type": "CARGO",
                    "voyage_id": voyage,
                    "ground_truth_json": track(ex["target_lat"], ex["target_lon"]),
                    "unmitigated_pred_json": track(off_lat, off_lon),
                    "mitigated_pred_json": track(on_lat, on_lon),
                    "uncertainty_poly_json": band_polygon(on_lat, on_lon, radius) if mitigated else [],
                    "tier1_divergence_m": divergence,
                    "staleness_min": stale[(cond, w)][1],
                    "trust_score": trust,
                    "example_deferred_pct": round(100.0 * float((radius > SAFETY_RADIUS_M).mean()), 1) if mitigated else 0.0,
                })
    assert len(metrics) == len(trajectories) == 48
    return {
        "run": "dashboard_export",
        "source": "real MarineCadastre AIS (Sept 1-12, 2025), simulated feed corruption",
        "tables": "field names match backend/db.py: experiment_metrics and vessel_trajectories",
        "field_notes": __doc__.split("Field notes (also written into the file):")[1].strip(),
        "safety_radius_m": SAFETY_RADIUS_M,
        "experiment_metrics": metrics,
        "vessel_trajectories": trajectories,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --- optional: load into the app's DuckDB -------------------------------------------------
METRIC_COLS = ["condition", "mitigated", "window_id", "displacement_mae_m", "tail_95_error_m", "data_discarded_pct",
               "decisions_deferred_pct", "tier2_corrupted_volume_pct", "avg_data_staleness_min", "model_compute_latency_ms"]
TRAJ_COLS = ["condition", "mitigated", "window_id", "mmsi", "vessel_name", "vessel_type", "ground_truth_json",
             "unmitigated_pred_json", "mitigated_pred_json", "uncertainty_poly_json", "tier1_divergence_m",
             "staleness_min", "trust_score"]


def load_into(db_path: Path, export: dict) -> None:
    """Replace the demo rows in the app's experiment_metrics and vessel_trajectories tables."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS experiment_metrics (condition VARCHAR, mitigated BOOLEAN,
            window_id INTEGER, displacement_mae_m DOUBLE, tail_95_error_m DOUBLE, data_discarded_pct DOUBLE,
            decisions_deferred_pct DOUBLE, tier2_corrupted_volume_pct DOUBLE, avg_data_staleness_min DOUBLE,
            model_compute_latency_ms DOUBLE)""")
        con.execute("""CREATE TABLE IF NOT EXISTS vessel_trajectories (condition VARCHAR, mitigated BOOLEAN,
            window_id INTEGER, mmsi VARCHAR, vessel_name VARCHAR, vessel_type VARCHAR, ground_truth_json VARCHAR,
            unmitigated_pred_json VARCHAR, mitigated_pred_json VARCHAR, uncertainty_poly_json VARCHAR,
            tier1_divergence_m DOUBLE, staleness_min DOUBLE, trust_score DOUBLE)""")
        con.execute("BEGIN TRANSACTION")
        con.execute("DELETE FROM experiment_metrics")
        con.execute("DELETE FROM vessel_trajectories")
        con.executemany(f"INSERT INTO experiment_metrics ({', '.join(METRIC_COLS)}) VALUES ({', '.join('?' * len(METRIC_COLS))})",
                        [[m[c] for c in METRIC_COLS] for m in export["experiment_metrics"]])
        con.executemany(f"INSERT INTO vessel_trajectories ({', '.join(TRAJ_COLS)}) VALUES ({', '.join('?' * len(TRAJ_COLS))})",
                        [[json.dumps(t[c]) if c.endswith("_json") else t[c] for c in TRAJ_COLS]
                         for t in export["vessel_trajectories"]])
        con.execute("COMMIT")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export real results for the dashboard.")
    parser.add_argument("--into", type=Path, help="also load into this app database (e.g. data/maritime_case.duckdb)")
    args = parser.parse_args()
    export = build()
    out = RESULTS / "dashboard_export_real.json"
    out.write_text(json.dumps(export, indent=2))
    m = pd.DataFrame(export["experiment_metrics"])
    print(m[m["window_id"] == 6][["condition", "mitigated", "displacement_mae_m", "data_discarded_pct",
                                  "decisions_deferred_pct", "tier2_corrupted_volume_pct", "avg_data_staleness_min",
                                  "model_compute_latency_ms"]].to_string(index=False))
    print(f"\n{len(export['experiment_metrics'])} metric rows, {len(export['vessel_trajectories'])} trajectory rows")
    print(f"Saved: {out.relative_to(REPO_ROOT)}")
    if args.into:
        load_into(args.into, export)
        print(f"Loaded into {args.into} (demo rows replaced)")


if __name__ == "__main__":
    main()
