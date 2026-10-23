"""Step 7: does the countermeasure backfire? (Requirement 7: failures, tradeoffs, boundaries)

Run from the repository root, AFTER rolling.py and summary.py:
    uv run python -m backend.ml.backfire

Three checks:

 A. COST ON CLEAN DATA. With no poisoning, the gate should barely matter. (From summary.py.)

 B. REAL SHARP TURNS FLAGGED AS "NOISE"? On clean data, every gate flag is a false alarm.
    We group clean fast-feed pings by how fast the ship is turning and compare false-alarm
    rates. Then we check whether predictions for turning ships got worse with the gate on.

 C. A DIFFERENT KIND OF BAD DATA. The gate was never tuned on any corruption (its limits
    come from clean week W1), so it should work on corruption it has never seen. We build
    two NEW corruptions in memory (the database is not changed) on top of the clean data:
      SHORT_STALE: repeats the report from only 3 pings (~3.5 min) earlier instead of 10,
                   so the ship looks only slightly behind. Harder to spot.
      GPS_BIAS:    a growing share of voyages (10% in W2 -> 40% in W5) have every fast-feed
                   ping shifted 300 m in one fixed direction (a mis-set receiver).
    For each: how much the gate catches, and W6 error with the gate off vs on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.ml.data import REPO_ROOT, load_pings
from backend.ml.features import build_features, make_dataset
from backend.ml.gate import apply_gate, compare_to_anchor, learn_limits, score_trust
from backend.ml.metrics import offset_m_to_latlon
from backend.ml.model_store import MODELS_DIR, load_models
from backend.ml.rolling import model_name, train_on_buffer, training_buffer
from backend.ml.train_baseline import errors_m, predict_offsets, score

RESULTS = REPO_ROOT / "results"
SEED = 42
STRESS_SEED = 7
GROUP_NAMES = {0: "cargo", 1: "tanker", 2: "tug", 3: "other"}
TURN_BINS = [-0.1, 5, 20, 60, np.inf]
TURN_LABELS = ["steady (<5 deg/min)", "gentle (5-20)", "sharp (20-60)", "very sharp (>60)"]
STALE_RATES = {2: 0.15, 3: 0.30, 4: 0.50, 5: 0.70}  # same growth as Person 1's STALE condition
BIAS_VOYAGE_SHARE = {2: 0.10, 3: 0.20, 4: 0.30, 5: 0.40}
BIAS_M = 300.0


def turning_bucket(df: pd.DataFrame) -> pd.Series:
    """How fast the ship is turning; slow ships (< 3 kn) get their own bucket (heading is unreliable)."""
    b = pd.cut(df["d_cog_per_min"].abs(), TURN_BINS, labels=TURN_LABELS).astype(object)
    b[df["sog"] < 3] = "slow (<3 kn)"
    return b.fillna("first ping (no turn rate)")


# --- B: sharp turns ------------------------------------------------------------------
def check_turns() -> dict:
    clean = load_pings("real", "NOMINAL")
    checked = compare_to_anchor(clean)
    limits = learn_limits(checked[checked["window_id"] == 1])
    scored = score_trust(checked, limits)
    turn = build_features(clean)[["voyage_id", "timestamp", "d_cog_per_min"]]
    fast = scored[scored["verifiable"]].merge(turn, on=["voyage_id", "timestamp"], how="left", validate="one_to_one")
    fast["turning"] = turning_bucket(fast)
    fast["ship"] = fast["vessel_group"].map(GROUP_NAMES)
    fa = (fast[fast["ship"].isin(["cargo", "tug"])]
          .groupby(["turning", "ship"])["suspect"].agg(["mean", "size"]).reset_index())
    false_alarms = {f"{r.ship} | {r.turning}": {"false_alarm_pct": round(100 * r["mean"], 2), "pings": int(r["size"])}
                    for _, r in fa.iterrows()}

    # Did predictions for turning ships get worse with the gate on? (clean data, W6, saved models)
    test = make_dataset(load_pings("real", "NOMINAL", windows=[6]))
    test["turning"] = turning_bucket(test)
    test["ship"] = test["vessel_group"].map(GROUP_NAMES)
    out = {}
    for cond in ("NOMINAL", "COMBINED"):
        for tag in ("unmitigated", "gated"):
            m = load_models(MODELS_DIR / f"{model_name(tag, cond, 'real', 6)}.joblib")["models"]
            test[f"{cond}_{tag}"] = errors_m(test, *predict_offsets(m, test))
    for (ship, turning), d in test[test["ship"].isin(["cargo", "tug"])].groupby(["ship", "turning"]):
        out[f"{ship} | {turning}"] = {
            "predictions": len(d),
            **{f"{c}_{t}_mae_m": round(float(d[f"{c}_{t}"].mean()), 1)
               for c in ("NOMINAL", "COMBINED") for t in ("unmitigated", "gated")},
        }
    return {"false_alarms_on_clean_data": false_alarms, "w6_error_by_turning": out}


# --- C: new kinds of bad data ------------------------------------------------------
def make_short_stale(clean: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    df = clean.sort_values(["voyage_id", "timestamp"]).reset_index(drop=True)
    earlier = df.groupby("voyage_id", sort=False)[["lat", "lon", "sog", "cog"]].shift(3)
    corrupted = np.zeros(len(df), bool)
    for w, rate in STALE_RATES.items():
        idx = np.flatnonzero(((df["window_id"] == w) & (df["source_tier"] == "TIER2_FAST")
                              & earlier["lat"].notna()).to_numpy())
        pick = rng.choice(idx, size=int(rate * idx.size), replace=False)
        corrupted[pick] = True
    for c in ["lat", "lon", "sog", "cog"]:
        df.loc[corrupted, c] = earlier.loc[corrupted, c]
    return df, corrupted


def make_gps_bias(clean: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    df = clean.sort_values(["voyage_id", "timestamp"]).reset_index(drop=True)
    corrupted = np.zeros(len(df), bool)
    for w, share in BIAS_VOYAGE_SHARE.items():
        voyages = df.loc[df["window_id"] == w, "voyage_id"].unique()
        chosen = rng.choice(voyages, size=int(share * voyages.size), replace=False)
        for v in chosen:
            rows = np.flatnonzero(((df["voyage_id"] == v) & (df["source_tier"] == "TIER2_FAST")).to_numpy())
            direction = rng.uniform(0, 2 * np.pi)
            lat, lon = offset_m_to_latlon(df.loc[rows, "lat"], df.loc[rows, "lon"],
                                          BIAS_M * np.cos(direction), BIAS_M * np.sin(direction))
            df.loc[rows, "lat"], df.loc[rows, "lon"] = lat, lon
            corrupted[rows] = True
    return df, corrupted


def check_new_corruption(name: str, pings: pd.DataFrame, corrupted: np.ndarray, test: pd.DataFrame) -> dict:
    retrain_w2 = make_dataset(load_pings("real", "NOMINAL", windows=[2]))["timestamp"].min()
    limits = learn_limits(compare_to_anchor(training_buffer(pings, 2, retrain_w2)))

    # How much does the gate catch? (graded with our own in-memory labels)
    scored = score_trust(compare_to_anchor(pings), limits)
    labels = pings.assign(corrupted=corrupted)[["voyage_id", "timestamp", "corrupted"]]
    g = scored.merge(labels, on=["voyage_id", "timestamp"], validate="one_to_one")
    fast = g[(g["source_tier"] == "TIER2_FAST") & g["window_id"].between(2, 5)]
    caught = float(fast.loc[fast["corrupted"], "suspect"].mean())
    false_alarm = float(fast.loc[~fast["corrupted"], "suspect"].mean())

    # Does the model still recover? Retrain on W1-W5 with the gate off and on, test on clean W6.
    retrain_at = test["timestamp"].min()
    buffer = training_buffer(pings[pings["window_id"] <= 5], 6, retrain_at)
    kept, _ = apply_gate(buffer, limits)
    result = {}
    for label, data in (("gate_off", buffer), ("gate_on", kept)):
        models, _ = train_on_buffer(data, retrain_at, SEED)
        s = score(test, *predict_offsets(models, test))
        result[f"{label}_mae_m"] = round(s["displacement_mae_m"], 1)
        result[f"{label}_cargo_mae_m"] = round(s["by_vessel_group"]["cargo"]["displacement_mae_m"], 1)
    return {
        "corrupted_fast_pings_W2_W5_pct": round(100 * float(fast["corrupted"].mean()), 1),
        "gate_catch_rate_pct": round(100 * caught, 1),
        "gate_false_alarm_pct": round(100 * false_alarm, 1),
        "data_discarded_pct_at_W6_retrain": round(100 * (1 - len(kept) / len(buffer)), 1),
        **result,
    }


def main() -> None:
    summary = json.loads((RESULTS / "experiment_summary_real.json").read_text())
    runs = {(r["condition"], r["countermeasure"] != "none"): r for r in summary["eight_runs"]}
    cargo_gaps = summary["uncertainty_check"]["cargo"]["gaps"]["NOMINAL"]
    all_gaps = summary["uncertainty_check"]["all_ships"]["gaps"]["NOMINAL"]
    cost = {
        "clean_gate_off_mae_m": runs[("NOMINAL", False)]["avg_error_m"],
        "clean_gate_on_mae_m": runs[("NOMINAL", True)]["avg_error_m"],
        "clean_gate_off_cargo_mae_m": runs[("NOMINAL", False)]["cargo_avg_error_m"],
        "clean_gate_on_cargo_mae_m": runs[("NOMINAL", True)]["cargo_avg_error_m"],
        "cost_all_ships_95pct_range_m": list(all_gaps.values())[0],
        "cost_cargo_95pct_range_m": list(cargo_gaps.values())[0],
        "clean_data_thrown_out_pct": runs[("NOMINAL", True)]["data_discarded_pct"],
    }
    print("A. Cost of the gate on clean data (W6)")
    print(f"   all ships: {cost['clean_gate_off_mae_m']} -> {cost['clean_gate_on_mae_m']} m "
          f"(95% range of the change: {cost['cost_all_ships_95pct_range_m']} m)")
    print(f"   cargo:     {cost['clean_gate_off_cargo_mae_m']} -> {cost['clean_gate_on_cargo_mae_m']} m "
          f"(95% range of the change: {cost['cost_cargo_95pct_range_m']} m)")

    turns = check_turns()
    print("\nB1. False alarms on clean data by how fast the ship is turning (fast-feed pings)")
    for k, v in turns["false_alarms_on_clean_data"].items():
        print(f"   {k:40} {v['false_alarm_pct']:6.2f}%  ({v['pings']:,} pings)")
    print("\nB2. W6 average error (m) by turning: clean data gate off/on, and poisoned 'Both' gate off/on")
    for k, v in turns["w6_error_by_turning"].items():
        print(f"   {k:40} n={v['predictions']:>5} | clean {v['NOMINAL_unmitigated_mae_m']:6.0f} -> "
              f"{v['NOMINAL_gated_mae_m']:6.0f} | both {v['COMBINED_unmitigated_mae_m']:6.0f} -> {v['COMBINED_gated_mae_m']:6.0f}")

    clean = load_pings("real", "NOMINAL", windows=[1, 2, 3, 4, 5])
    test = make_dataset(load_pings("real", "NOMINAL", windows=[6]))
    rng = np.random.default_rng(STRESS_SEED)
    new = {}
    for name, maker in (("SHORT_STALE", make_short_stale), ("GPS_BIAS", make_gps_bias)):
        pings, corrupted = maker(clean, rng)
        new[name] = check_new_corruption(name, pings, corrupted, test)
    clean_w6 = runs[("NOMINAL", False)]
    print(f"\nC. New kinds of bad data the gate never saw (W6; clean data gate off = "
          f"{clean_w6['avg_error_m']} m all, {clean_w6['cargo_avg_error_m']} m cargo)")
    for name, r in new.items():
        print(f"   {name:12} corrupted {r['corrupted_fast_pings_W2_W5_pct']:4.1f}% | caught {r['gate_catch_rate_pct']:5.1f}%"
              f" | false alarms {r['gate_false_alarm_pct']:4.1f}% | all ships {r['gate_off_mae_m']:.0f} -> "
              f"{r['gate_on_mae_m']:.0f} m | cargo {r['gate_off_cargo_mae_m']:.0f} -> {r['gate_on_cargo_mae_m']:.0f} m")

    out = RESULTS / "backfire_checks_real.json"
    out.write_text(json.dumps({
        "run": "backfire_checks",
        "A_cost_on_clean_data": cost,
        "B_sharp_turns": turns,
        "C_new_corruptions": {"settings": {"short_stale_lag_pings": 3, "stale_rates": STALE_RATES,
                                           "gps_bias_m": BIAS_M, "bias_voyage_share": BIAS_VOYAGE_SHARE,
                                           "seed": STRESS_SEED, "note": "built in memory; database unchanged"},
                              "results": new},
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2, default=str))
    print(f"\nSaved: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
