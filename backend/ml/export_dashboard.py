"""Step 8: export real results in the exact format the dashboard reads.

Run from the repository root, AFTER rolling.py:
    uv run python -m backend.ml.export_dashboard

Output: results/dashboard_export_real.json. The backend (backend/db.py) loads this file
automatically when it starts, so every copy of the site, local or hosted, shows these
real numbers. Commit the file after re-running this script.

Two lists, with field names matching backend/db.py:

  experiment_metrics   one row per (condition, gate off/on, window W1-W6) = 48 rows
  vessel_trajectories  same keys x 2 example ships (vessel_type presets) = 96 rows

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
  example ships (fixed rules, not picked for looking good), per test window:
    CARGO_CONTAINER           the cargo voyage that traveled farthest (first to last
                              position) among voyages with >= 15 predictions
    HARBOR_TUG_RARE_MANEUVER  the tug job (<= 6 h, >= 10 sharp-turn reports of >= 20 deg/min)
                              with the highest share of those valid reports wrongly flagged by
                              the gate on CLEAN data (the Requirement 7 failure case)
  tracks                      ground truth = true position 15 min after each ping;
                              predictions = predicted position 15 min after each ping
                              (each point is a separate 15-min forecast). Tugs: only the
                              30 minutes around their sharpest turn are drawn
  uncertainty polygon         a band around the gate-on prediction, as wide as each
                              prediction's 90% uncertainty radius (for tracks with a turn
                              > 60 deg: one outline around all the uncertainty circles, since
                              a band would fold over itself); empty with the gate off
  trust / divergence          what the gate sees in that ship's fast-feed pings for the
                              selected condition and window (median distance from the
                              anchor track, mean trust score)
  example_* fields            that ship's own numbers: average and 95th-percentile error,
                              % of its fast-feed pings the gate flags, % of its
                              predictions handed to a human
  max_turn_deg_per_min        fastest turn in that voyage (clean data, ship >= 3 kn)
  clean_false_alarm_pct       % of that ship's valid fast-feed pings the gate flags on
                              CLEAN data (all of these are false alarms)
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.ml.data import CONDITIONS, REPO_ROOT, connect, load_pings
from backend.ml.features import build_features, make_dataset
from backend.ml.gate import compare_to_anchor, learn_limits, score_trust
from backend.ml.metrics import haversine_m, offset_m_to_latlon
from backend.ml.model_store import MODELS_DIR, load_models
from backend.ml.rolling import TEST_WINDOWS, model_name, training_buffer
from backend.ml.train_baseline import errors_m, lgbm_params, predict_offsets, split_by_voyage_in_time
from backend.ml.uncertainty import SAFETY_RADIUS_M, predict_range, train_uncertainty_models, uncertainty_radius_m

RESULTS = REPO_ROOT / "results"
MOD_LOG = REPO_ROOT / "backend" / "database" / "modification_log.json"
STALE_LAG_PINGS = 10  # Person 1's stale rule (backend/database/build_database.py)
MAX_TRACK_POINTS = 40
SEED = 42
MIN_EXAMPLE_PREDICTIONS = 15
SHARP_TURN_DEG_PER_MIN = 20.0
MIN_SPEED_FOR_TURN_KN = 3.0  # heading is unreliable for nearly stopped ships
TUG_MIN_SHARP_REPORTS = 10
TUG_MAX_HOURS = 6.0
TUG_FOCUS_MINUTES = 30  # map shows this much time around the tug's sharpest turn
PRESETS = {"CARGO_CONTAINER": 0, "HARBOR_TUG_RARE_MANEUVER": 2}  # dashboard preset -> vessel_group
CONDITION_TEXT = {"NOMINAL": "Clean feeds", "STALE": "Stale poisoning", "LOW_QUALITY": "Low-quality poisoning",
                  "COMBINED": "Combined poisoning"}


# --- helpers -------------------------------------------------------------------------
def _eligible(test: pd.DataFrame, group: int) -> pd.DataFrame:
    """Rows of voyages of this ship group with enough predictions to draw a track."""
    rows = test[test["vessel_group"] == group]
    counts = rows.groupby("voyage_id")["voyage_id"].transform("size")
    return rows[counts >= MIN_EXAMPLE_PREDICTIONS]


def pick_cargo_voyage(test: pd.DataFrame) -> str:
    """The cargo voyage that traveled farthest, first to last position (ties: smallest voyage_id)."""
    rows = _eligible(test, PRESETS["CARGO_CONTAINER"]).sort_values(["voyage_id", "timestamp"])
    ends = rows.groupby("voyage_id").agg(lat0=("lat", "first"), lon0=("lon", "first"),
                                         lat1=("lat", "last"), lon1=("lon", "last"))
    ends["distance_m"] = haversine_m(ends["lat0"], ends["lon0"], ends["lat1"], ends["lon1"])
    return ends.reset_index().sort_values(["distance_m", "voyage_id"], ascending=[False, True]).iloc[0]["voyage_id"]


def pick_tug_voyage(test: pd.DataFrame, clean_flags: pd.DataFrame) -> str:
    """The tug job whose valid sharp-turn reports the gate most often wrongly flags on clean data.

    Among tug voyages with >= 15 predictions, >= 10 sharp-turn reports and lasting <= 6 hours
    (a typical tug job; multi-day tracks would win on volume alone): the highest SHARE of
    sharp-turn reports flagged. Ties: more flags, then smallest voyage_id. clean_flags = gate
    decisions on CLEAN fast-feed pings, so every flag is a false alarm.
    """
    rows = _eligible(test, PRESETS["HARBOR_TUG_RARE_MANEUVER"])
    hours = rows.groupby("voyage_id")["timestamp"].agg(lambda t: (t.max() - t.min()).total_seconds() / 3600)
    f = clean_flags[clean_flags["voyage_id"].isin(hours.index) & clean_flags["sharp_turn"]]
    score = f.groupby("voyage_id")["suspect"].agg(flags="sum", sharp="size").join(hours.rename("hours"))
    pool = score[(score["sharp"] >= TUG_MIN_SHARP_REPORTS) & (score["hours"] <= TUG_MAX_HOURS)]
    if pool.empty:  # fall back to any tug with a sharp turn
        pool = score
    pool = pool.assign(share=pool["flags"] / pool["sharp"]).reset_index()
    return pool.sort_values(["share", "flags", "voyage_id"], ascending=[False, False, True]).iloc[0]["voyage_id"]


def clean_gate_flags(limits: dict) -> pd.DataFrame:
    """Gate decisions on CLEAN fast-feed pings (all windows) plus each ping's turn rate."""
    clean = load_pings("real", "NOMINAL")
    scored = score_trust(compare_to_anchor(clean), limits)
    fast = scored[scored["verifiable"]][["voyage_id", "timestamp", "suspect"]]
    turn = build_features(clean)[["voyage_id", "timestamp", "d_cog_per_min", "sog"]]
    out = fast.merge(turn, on=["voyage_id", "timestamp"], how="left", validate="one_to_one")
    out["sharp_turn"] = (out["d_cog_per_min"].abs() >= SHARP_TURN_DEG_PER_MIN) & (out["sog"] >= MIN_SPEED_FOR_TURN_KN)
    return out


def max_turn_rate(voyage_rows: pd.DataFrame) -> float:
    """Fastest turn (deg/min) in a voyage's clean data, ignoring nearly stopped moments."""
    moving = voyage_rows[voyage_rows["sog"] >= MIN_SPEED_FOR_TURN_KN]
    return round(float(moving["d_cog_per_min"].abs().max()), 1) if len(moving) else 0.0


def thin(idx: np.ndarray, k: int = MAX_TRACK_POINTS) -> np.ndarray:
    """At most k evenly spaced positions, always keeping the first and last."""
    return idx if len(idx) <= k else idx[np.unique(np.linspace(0, len(idx) - 1, k).round().astype(int))]


def _segs_intersect(p1: tuple[float, float], p2: tuple[float, float],
                    p3: tuple[float, float], p4: tuple[float, float]) -> bool:
    def ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def _prune_polygon_loops(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove any self-crossing loops in a closed polygon ring so it is strictly simple."""
    out = list(pts)
    changed = True
    while changed and len(out) >= 4:
        changed = False
        m = len(out)
        for i in range(m):
            a, b = out[i], out[(i + 1) % m]
            for j in range(i + 2, m):
                if i == 0 and j == m - 1:
                    continue
                c, d = out[j], out[(j + 1) % m]
                if _segs_intersect(a, b, c, d):
                    arc1_len = j - i
                    arc2_len = m - arc1_len
                    out = out[:i + 1] + out[j + 1:] if arc1_len <= arc2_len else out[i + 1:j + 1]
                    changed = True
                    break
            if changed:
                break
    return out


def band_polygon(lat: np.ndarray, lon: np.ndarray, radius_m: np.ndarray) -> list[list[float]]:
    """A smooth, non-crossing band around a track: each point pushed left and right by its radius."""
    n = len(lat)
    if n < 2:
        return []
    if n >= 5:
        k = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
        c_lat = np.convolve(np.pad(lat, 2, mode="edge"), k, mode="valid")
        c_lon = np.convolve(np.pad(lon, 2, mode="edge"), k, mode="valid")
        c_lat[0], c_lon[0] = lat[0], lon[0]
        c_lat[-1], c_lon[-1] = lat[-1], lon[-1]
    else:
        c_lat, c_lon = lat, lon

    step = min(3, n - 1)
    i_prev, i_next = np.maximum(np.arange(n) - step, 0), np.minimum(np.arange(n) + step, n - 1)
    d_north = (c_lat[i_next] - c_lat[i_prev]) * 111_195.0
    d_east = (c_lon[i_next] - c_lon[i_prev]) * 111_195.0 * np.cos(np.radians(c_lat))
    norm = np.hypot(d_north, d_east)
    norm[norm == 0] = 1.0
    left_n, left_e = -d_east / norm, d_north / norm  # unit vector to the left of travel
    r_smooth = np.convolve(np.pad(radius_m, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    l_lat, l_lon = offset_m_to_latlon(c_lat, c_lon, left_n * r_smooth, left_e * r_smooth)
    r_lat, r_lon = offset_m_to_latlon(c_lat, c_lon, -left_n * r_smooth, -left_e * r_smooth)
    pts = _prune_polygon_loops(list(zip(l_lat, l_lon)) + list(zip(r_lat[::-1], r_lon[::-1])))
    return [[round(float(a), 6), round(float(b), 6)] for a, b in pts]


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points (Andrew's monotone chain), counter-clockwise, no repeats."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def uncertainty_outline(lat: np.ndarray, lon: np.ndarray, radius_m: np.ndarray,
                        preset: str = "CARGO_CONTAINER") -> list[list[float]]:
    """The uncertainty area around a predicted track, as one simple (non-crossing) polygon.

    Cargo transit tracks follow a curved band along the channel. Harbor tug maneuvers stay in a
    tight harbor basin (radius > track length), so they use the outline around their uncertainty circles.
    """
    if len(lat) < 2:
        return []
    if preset != "HARBOR_TUG_RARE_MANEUVER":
        return band_polygon(lat, lon, radius_m)
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    north = (lat - lat0) * 111_195.0
    east = (lon - lon0) * 111_195.0 * np.cos(np.radians(lat0))
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circles = np.concatenate([
        np.column_stack([n + r * np.cos(angles), e + r * np.sin(angles)]) for n, e, r in zip(north, east, radius_m)
    ])
    hull = _convex_hull(circles)
    h_lat, h_lon = offset_m_to_latlon(lat0, lon0, hull[:, 0], hull[:, 1])
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(h_lat, h_lon)]


def focus_rows(full: pd.DataFrame, preset: str) -> pd.DataFrame:
    """Rows to draw on the map. Tugs: the 30 minutes around their sharpest turn (the failure
    moment), because a whole hour of harbor work drawn as one line is unreadable."""
    if preset != "HARBOR_TUG_RARE_MANEUVER":
        return full
    moving = full[full["sog"] >= MIN_SPEED_FOR_TURN_KN]
    if moving.empty:
        return full
    center = moving.loc[moving["d_cog_per_min"].abs().idxmax(), "timestamp"]
    window = full[(full["timestamp"] - center).abs() <= pd.Timedelta(minutes=TUG_FOCUS_MINUTES / 2)]
    return window if len(window) >= 5 else full


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


@lru_cache(maxsize=None)
def _window_pings(cond: str, window: int) -> pd.DataFrame:
    """One condition's pings for one window (cached: each is used by several example ships)."""
    return load_pings("real", cond, windows=[window])


def gate_view(cond: str, window: int, voyage: str, limits: dict) -> tuple[float, float, float]:
    """What the gate sees in this voyage's fast-feed pings for one condition and window:
    median distance from the anchor track (m), mean trust score, % of pings flagged."""
    pings = _window_pings(cond, window)
    pings = pings[pings["voyage_id"] == voyage]
    scored = score_trust(compare_to_anchor(pings), limits)
    fast = scored[scored["verifiable"]]
    if fast.empty:
        return 0.0, 1.0, 0.0
    return (round(float(fast["anchor_dist_m"].median()), 1), round(float(fast["trust_score"].mean()), 3),
            round(100.0 * float(fast["suspect"].mean()), 1))


def explain(preset: str, cond: str, w: int, name: str, view: tuple, err: dict, newest: tuple,
            turn: float, false_alarm: float) -> str:
    """One sentence built only from this ship's real numbers."""
    tag = {1: "W1 (Reference Baseline)", 6: "W6 (Unseen Test Period)"}.get(w, f"W{w}")
    divergence, trust, flagged = view
    corrupted, staleness = newest
    errors = (f"Model error for this ship: {err[False]:.0f} m without the gate, {err[True]:.0f} m with it.")
    if preset == "HARBOR_TUG_RARE_MANEUVER":
        return (f"[{tag} // FAILURE MODE] {name} turns at up to {turn:.0f} deg/min. Even on clean data the trust "
                f"gate wrongly flags {false_alarm:.0f}% of its valid fast-feed reports, because it compares them "
                f"to a straight line between trusted reports ~6 min apart. {errors}")
    text = (f"[{tag}] {CONDITION_TEXT[cond]}: {name}'s fast-feed reports sit a median {divergence:.0f} m from the "
            f"trusted anchor track (trust {trust:.2f}); the gate flags {flagged:.0f}% of them.")
    if cond != "NOMINAL" and w > 1:
        text += (f" {corrupted:.0f}% of fast-feed reports in the newest training week were corrupted"
                 f" (est. average report age {staleness:.1f} min).")
    return f"{text} {errors}"


def _source_label() -> str:
    """Data description for the dashboard, taken from the manifest (never hard-coded)."""
    manifest = json.loads((REPO_ROOT / "backend" / "database" / "manifest.json").read_text())
    first, last = (d[:10] for d in manifest["date_range_utc"])
    return f"real MarineCadastre AIS ({first} to {last}), simulated feed corruption"


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

    # Gate limits for the "what the gate sees" fields (learned from W1, as in rolling.py).
    retrain_w2 = tests[2]["timestamp"].min()
    limits = {c: learn_limits(compare_to_anchor(training_buffer(load_pings("real", c, windows=[1]), 2, retrain_w2)))
              for c in CONDITIONS}
    flags = clean_gate_flags(limits["NOMINAL"])
    false_alarm_pct = (100.0 * flags.groupby("voyage_id")["suspect"].mean()).round(1).to_dict()
    examples = {w: {"CARGO_CONTAINER": pick_cargo_voyage(t), "HARBOR_TUG_RARE_MANEUVER": pick_tug_voyage(t, flags)}
                for w, t in tests.items()}

    metrics, trajectories = [], []
    latency = {}
    for cond in CONDITIONS:
        for w in range(1, 7):
            if w == 1:
                bundles = {False: base, True: {**base, "uncertainty": base_range}}
            else:
                bundles = {m: load_models(MODELS_DIR / f"{model_name('gated' if m else 'unmitigated', cond, 'real', w)}.joblib")["models"]
                           for m in (False, True)}
            newest_week = 1 if w == 1 else w - 1  # newest week in the training data
            corrupted_pct, staleness = stale[(cond, newest_week)]

            for mitigated in (False, True):
                if mitigated not in latency:
                    latency[mitigated] = single_prediction_ms(bundles[mitigated], tests[w].iloc[[0]])
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

            for preset, voyage in examples[w].items():
                full = tests[w][tests[w]["voyage_id"] == voyage].sort_values("timestamp")
                shown = focus_rows(full, preset)
                ex = shown.iloc[thin(np.arange(len(shown)))]
                off_lat, off_lon = predicted_track(bundles[False], ex)
                on_lat, on_lon = predicted_track(bundles[True], ex)
                radius_ex = uncertainty_radius_m(predict_range(bundles[True]["uncertainty"], ex))
                radius_full = uncertainty_radius_m(predict_range(bundles[True]["uncertainty"], full))
                err = {m: errors_m(full, *predict_offsets(bundles[m], full)) for m in (False, True)}
                view = gate_view(cond, w, voyage, limits[cond])
                mmsi, vname = names.get(voyage, (voyage.split("_")[0], voyage))
                turn = max_turn_rate(full)
                fa = false_alarm_pct.get(voyage, 0.0)
                text = explain(preset, cond, w, vname, view, {m: float(e.mean()) for m, e in err.items()},
                               (corrupted_pct, staleness), turn, fa)
                for mitigated in (False, True):
                    trajectories.append({
                        "condition": cond, "mitigated": mitigated, "window_id": w,
                        "mmsi": mmsi, "vessel_name": vname, "vessel_type": preset,
                        "voyage_id": voyage,
                        "ground_truth_json": track(ex["target_lat"], ex["target_lon"]),
                        "unmitigated_pred_json": track(off_lat, off_lon),
                        "mitigated_pred_json": track(on_lat, on_lon),
                        "uncertainty_poly_json": uncertainty_outline(on_lat, on_lon, radius_ex, preset) if mitigated else [],
                        "tier1_divergence_m": view[0],
                        "staleness_min": stale[(cond, w)][1],
                        "trust_score": view[1],
                        "example_mae_m": round(float(err[mitigated].mean()), 1),
                        "example_tail_95_m": round(float(np.percentile(err[mitigated], 95)), 1),
                        "example_discarded_pct": view[2] if mitigated else 0.0,
                        "example_deferred_pct": round(100.0 * float((radius_full > SAFETY_RADIUS_M).mean()), 1) if mitigated else 0.0,
                        "example_predictions": int(len(full)),
                        "max_turn_deg_per_min": turn,
                        "clean_false_alarm_pct": fa,
                        "explanation": text,
                    })
    assert len(metrics) == 48 and len(trajectories) == 96, (len(metrics), len(trajectories))
    return {
        "run": "dashboard_export",
        "source": _source_label(),
        "tables": "field names match backend/db.py: experiment_metrics and vessel_trajectories",
        "examples_by_window": {f"W{w}": e for w, e in examples.items()},
        "field_notes": __doc__.split("Field notes (also written into the file):")[1].strip(),
        "safety_radius_m": SAFETY_RADIUS_M,
        "experiment_metrics": metrics,
        "vessel_trajectories": trajectories,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> None:
    export = build()
    out = RESULTS / "dashboard_export_real.json"
    out.write_text(json.dumps(export, indent=2))
    m = pd.DataFrame(export["experiment_metrics"])
    print(m[m["window_id"] == 6][["condition", "mitigated", "displacement_mae_m", "data_discarded_pct",
                                  "decisions_deferred_pct", "tier2_corrupted_volume_pct", "avg_data_staleness_min",
                                  "model_compute_latency_ms"]].to_string(index=False))
    t = pd.DataFrame(export["vessel_trajectories"])
    ships = (t[(t["condition"] == "COMBINED") & t["mitigated"]]
             [["window_id", "vessel_type", "vessel_name", "example_predictions", "max_turn_deg_per_min",
               "clean_false_alarm_pct", "example_mae_m"]])
    print(f"\nExample ships (COMBINED, gate on):\n{ships.to_string(index=False)}")
    print(f"\n{len(export['experiment_metrics'])} metric rows, {len(export['vessel_trajectories'])} trajectory rows")
    print(f"Saved: {out.relative_to(REPO_ROOT)}  (the backend loads it automatically on start; commit it)")


if __name__ == "__main__":
    main()
