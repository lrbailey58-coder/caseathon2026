"""Weekly retraining, with and without the trust gate: does the gate undo the poisoning?

Run from the repository root:
    uv run python -m backend.ml.rolling                       # all 4 conditions, gate off AND on
    uv run python -m backend.ml.rolling --conditions STALE    # just one condition (faster)

For each condition (NOMINAL, STALE, LOW_QUALITY, COMBINED) we replay the weeks:

    step 1: retrain on what arrived in W1      -> test on W2
    step 2: retrain on what arrived in W1-W2   -> test on W3
    ...
    step 5: retrain on what arrived in W1-W5   -> test on W6 (the final, unseen period)

Each step is run twice on the SAME training buffer and the SAME test cases:
    gate OFF ("naive retraining"): learn from everything that arrived, corrupted rows included.
    gate ON  (countermeasure):     the trust gate (gate.py) first drops suspect fast-feed pings,
                                   and uncertain predictions are handed to a human (uncertainty.py).

Error numbers (displacement_mae_m, tail_95_error_m) are always over ALL test predictions,
deferred ones included, so handing work to a human can't make the AI look better.

The training buffer at each step = all pings (both feeds) from earlier windows with a
timestamp before the moment of retraining (the test window's first ping). The slow
anchor feed arrives ANCHOR_DELAY_MIN late, so anchor pings newer than that are not in
the buffer yet. The gate's "normal" limits are learned from week W1 only.

Testing always uses CLEAN data (NOMINAL inputs, true positions), so every run is
scored on exactly the same cases and differences come only from training
(training-time poisoning). Training rows must have their 15-min answer before the
moment of retraining.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.ml.data import CONDITIONS, REPO_ROOT, defect_share_by_window, load_pings
from backend.ml.features import HORIZON_S, dead_reckoning_offsets, make_dataset
from backend.ml.gate import apply_gate, compare_to_anchor, learn_limits
from backend.ml.model_store import save_models
from backend.ml.train_baseline import errors_m, lgbm_params, predict_offsets, score, train_models
from backend.ml.uncertainty import SAFETY_RADIUS_M, evaluate_deferral, predict_range, train_uncertainty_models

TEST_WINDOWS = [2, 3, 4, 5, 6]
ANCHOR_DELAY_MIN = 45  # the slow, accurate feed arrives 30-60 min late (Architecture.md)


def model_name(tag: str, condition: str, source: str, test_window: int) -> str:
    """File name (without .joblib) of a saved step model, e.g. rolling_gated_combined_real_w6."""
    return f"rolling_{tag}_{condition.lower()}_{source}_w{test_window}"


def training_buffer(pings: pd.DataFrame, test_window: int, retrain_at: pd.Timestamp) -> pd.DataFrame:
    """Everything that had ARRIVED by the moment of retraining."""
    earlier = (pings["window_id"] < test_window) & (pings["timestamp"] < retrain_at)
    anchor_not_arrived = (pings["source_tier"] == "TIER1_ANCHOR") & (
        pings["timestamp"] > retrain_at - pd.Timedelta(minutes=ANCHOR_DELAY_MIN))
    return pings[earlier & ~anchor_not_arrived].reset_index(drop=True)


def train_on_buffer(buffer: pd.DataFrame, retrain_at: pd.Timestamp, seed: int) -> tuple[dict, pd.DataFrame]:
    """Turn a buffer of pings into training rows (answers known before retraining) and train."""
    data = make_dataset(buffer)
    train = data[data["timestamp"] + pd.Timedelta(seconds=HORIZON_S) <= retrain_at].reset_index(drop=True)
    return train_models(train, seed), train


def run_rolling(conditions: list[str], seed: int = 42, source: str = "real") -> dict:
    # Clean test sets, built once and shared by every run.
    clean_tests = {w: make_dataset(load_pings(source, "NOMINAL", windows=[w], seed=seed)) for w in TEST_WINDOWS}
    physics = {w: score(t, *dead_reckoning_offsets(t)) for w, t in clean_tests.items()}

    rows = []
    for condition in conditions:
        t0 = time.perf_counter()
        pings = load_pings(source, condition, windows=[1, 2, 3, 4, 5], seed=seed)
        defects = defect_share_by_window(condition).set_index("window_id")

        # The gate's limits come from week W1 only (known clean), using pings that had
        # arrived by the first retraining.
        first_retrain = clean_tests[TEST_WINDOWS[0]]["timestamp"].min()
        w1 = training_buffer(pings, 2, first_retrain)
        limits = learn_limits(compare_to_anchor(w1))

        for test_w in TEST_WINDOWS:
            test = clean_tests[test_w]
            retrain_at = test["timestamp"].min()
            buffer = training_buffer(pings, test_w, retrain_at)
            kept, scored = apply_gate(buffer, limits)
            newest = scored["window_id"] == test_w - 1

            for mitigated, train_pings in ((False, buffer), (True, kept)):
                models, train = train_on_buffer(train_pings, retrain_at, seed)
                start = time.perf_counter()
                north, east = predict_offsets(models, test)
                predict_ms = (time.perf_counter() - start) * 1000
                ai = score(test, north, east)
                deferral = None
                if mitigated:
                    range_models = train_uncertainty_models(train, lgbm_params(seed))
                    deferral = evaluate_deferral(test, errors_m(test, north, east), predict_range(range_models, test))

                buf = defects.loc[defects.index < test_w]  # REPORTING ONLY, never given to the model
                rows.append({
                    "condition": condition,
                    "mitigated": mitigated,
                    "train_windows": f"W1-W{test_w - 1}",
                    "window_id": test_w,  # the window being predicted
                    "buffer_pings": len(buffer),
                    "n_train_rows": len(train),
                    "n_train_voyages": int(train["voyage_id"].nunique()),
                    "n_test_rows": len(test),
                    "displacement_mae_m": ai["displacement_mae_m"],
                    "tail_95_error_m": ai["tail_95_error_m"],
                    "by_vessel_group": ai["by_vessel_group"],
                    "physics_mae_m": physics[test_w]["displacement_mae_m"],
                    "physics_tail_95_m": physics[test_w]["tail_95_error_m"],
                    "data_discarded_pct": round(100.0 * (1 - len(kept) / len(buffer)), 2) if mitigated else 0.0,
                    "newest_window_discarded_pct": (
                        round(100.0 * scored.loc[newest, "suspect"].mean(), 2) if mitigated else 0.0),
                    "decisions_deferred_pct": deferral["decisions_deferred_pct"] if mitigated else 0.0,
                    "deferral": deferral,
                    "train_buffer_corrupted_pct": round(100.0 * buf["corrupted_rows"].sum() / buf["rows"].sum(), 2),
                    "predict_total_ms": round(predict_ms, 2),
                    "predict_ms_per_row": round(predict_ms / len(test), 5),
                })
                gate = "ON " if mitigated else "OFF"
                print(f"  {condition:12} gate {gate} train W1-W{test_w - 1} ({len(train):>7,} rows) -> test W{test_w}:"
                      f" {ai['displacement_mae_m']:6.0f} m avg | discarded {rows[-1]['data_discarded_pct']:5.1f}%")

                # Save every step's models (used by summary.py, backfire.py, export_dashboard.py).
                tag = "gated" if mitigated else "unmitigated"
                bundle = {**models, "uncertainty": range_models} if mitigated else models
                save_models(bundle, model_name(tag, condition, source, test_w), info={
                    "condition": condition, "mitigated": mitigated, "train_windows": f"W1-W{test_w - 1}",
                    "test_window": test_w, "source": source, "seed": seed, "gate_limits": limits,
                    "safety_radius_m": SAFETY_RADIUS_M if mitigated else None})
        print(f"  {condition} done in {time.perf_counter() - t0:.0f} s")

    return {
        "run": "rolling_retrain_gate_off_vs_on",
        "data_source": source,
        "seed": seed,
        "conditions": conditions,
        "design": {
            "steps": "retrain on what arrived in W1..Wk -> test W(k+1), for k = 1..5",
            "buffer": f"both feeds, earlier windows, timestamp before retraining; anchor feed delayed {ANCHOR_DELAY_MIN} min",
            "gate_off": "train on the whole buffer (naive retraining)",
            "gate_on": "trust gate drops suspect fast-feed pings first; limits learned from W1 only; "
                       f"predictions with a 90% range radius > {SAFETY_RADIUS_M:.0f} m are deferred to a human",
            "error_metric_scope": "all test predictions, deferred ones included",
            "test_data": "clean NOMINAL inputs and true positions, identical for all runs",
            "poisoning_stage": "training-time (retraining buffer); test-time inputs are clean",
            "no_future_rule": "train rows need their 15-min answer before retraining",
            "window_rule": "voyage belongs to the window of its first ping",
        },
        "lgbm_params": lgbm_params(seed),
        "rows": rows,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _cargo_mae(groups: dict) -> float:
    return groups.get("cargo", {}).get("displacement_mae_m", np.nan)


def print_tables(result: dict) -> None:
    df = pd.DataFrame(result["rows"])
    df["cargo_mae_m"] = df["by_vessel_group"].map(_cargo_mae)
    conds = [c for c in CONDITIONS if c in set(df["condition"])]
    df["run"] = df["condition"].str.replace("LOW_QUALITY", "NOISY") + np.where(df["mitigated"], " on", " off")
    order = [f"{c.replace('LOW_QUALITY', 'NOISY')} {m}" for c in conds for m in ("off", "on")]

    for title, col in (("All ships, average error (m)", "displacement_mae_m"),
                       ("Cargo ships only, average error (m)", "cargo_mae_m")):
        t = df.pivot(index="window_id", columns="run", values=col)[order]
        if col == "displacement_mae_m":
            t["PHYSICS"] = df.groupby("window_id")["physics_mae_m"].first()
        t.index = [f"test W{w}" for w in t.index]
        print(f"\n{title}  (off = no gate, on = trust gate)\n{t.round(0).astype(int).to_string()}")

    t = df[df["mitigated"]].pivot(index="window_id", columns="condition", values="data_discarded_pct")[conds]
    t.index = [f"test W{w}" for w in t.index]
    print(f"\nData thrown out by the gate (% of training buffer)\n{t.round(1).to_string()}")

    on = df[df["mitigated"]]
    d = pd.DataFrame(list(on["deferral"]), index=on.index).join(on[["condition", "window_id"]])
    t = d.pivot(index="window_id", columns="condition", values="decisions_deferred_pct")[conds]
    t.index = [f"test W{w}" for w in t.index]
    print(f"\nPredictions handed to a human (%), safety radius {SAFETY_RADIUS_M:.0f} m\n{t.round(1).to_string()}")
    w6 = d[d["window_id"] == TEST_WINDOWS[-1]].set_index("condition").loc[conds]
    cols = ["decisions_deferred_pct", "accepted_mae_m", "deferred_mae_m", "worst_10pct_deferred_pct",
            "range_coverage_north_pct", "range_coverage_east_pct"]
    w6t = w6[cols].astype(float).round(1)
    w6t.columns = ["handed off %", "avg err kept (m)", "avg err handed off (m)", "worst 10% handed off %",
                   "range hit N %", "range hit E %"]
    print(f"\nW6 hand-off detail (gate on)\n{w6t.to_string()}")
    print("  handed off % by ship type:", {c: w6.loc[c, "deferred_pct_by_ship_type"] for c in conds})

    if "NOMINAL" in conds:
        last = df[df["window_id"] == TEST_WINDOWS[-1]].set_index("run")
        print(f"\nW6 recovery: how much of the poisoning damage the gate undid (vs. clean data, gate off)")
        for col, label in (("displacement_mae_m", "all ships"), ("cargo_mae_m", "cargo")):
            clean = last.loc["NOMINAL off", col]
            for c in conds[1:]:
                name = c.replace("LOW_QUALITY", "NOISY")
                off, on = last.loc[f"{name} off", col], last.loc[f"{name} on", col]
                damage = off - clean
                recovered = 100.0 * (off - on) / damage if damage > 5 else float("nan")
                print(f"  {label:9} {name:9}: clean {clean:5.0f} m | gate off {off:5.0f} m | gate on {on:5.0f} m"
                      f" | damage undone: {'n/a (damage < 5 m)' if np.isnan(recovered) else f'{recovered:.0f}%'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly retraining with the trust gate off and on.")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Rolling retraining, gate off vs on | real data | seed {args.seed}")
    result = run_rolling(args.conditions, seed=args.seed, source="real")
    out = REPO_ROOT / "results" / "rolling_real.json"
    out.write_text(json.dumps(result, indent=2))
    print_tables(result)
    print(f"\nSaved: {out.relative_to(REPO_ROOT)} (+ every step's models in models/)")


if __name__ == "__main__":
    main()
