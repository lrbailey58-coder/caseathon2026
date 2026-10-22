"""Weekly retraining WITHOUT the countermeasure: does poisoned data make the model worse over time?

Run from the repository root:
    uv run python -m backend.ml.rolling                       # all 4 conditions, real data
    uv run python -m backend.ml.rolling --conditions NOMINAL  # just one (faster)

For each condition (NOMINAL, STALE, LOW_QUALITY, COMBINED) we replay the weeks:

    step 1: train on W1          -> test on W2
    step 2: train on W1-W2       -> test on W3
    ...
    step 5: train on W1-W5       -> test on W6 (the final, unseen period)

Training ("naive retraining") uses EVERYTHING that arrived for that condition:
both feeds, corrupted rows included, with the 15-min answers also taken from that
same corrupted stream. That is how the poison gets in.

Testing always uses CLEAN data (NOMINAL inputs, true positions), so every
condition is scored on exactly the same test cases, and any difference comes
only from what the model learned in training (training-time poisoning).

No-future rule: a training row is only used if its 15-min answer was known
before the test window's first ping (windows overlap a little in time).
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
from backend.ml.model_store import save_models
from backend.ml.train_baseline import lgbm_params, predict_offsets, score, train_models

TEST_WINDOWS = [2, 3, 4, 5, 6]


def run_rolling(conditions: list[str], seed: int = 42, source: str = "real") -> dict:
    # Clean test sets, built once and shared by every condition.
    clean_tests = {w: make_dataset(load_pings(source, "NOMINAL", windows=[w], seed=seed)) for w in TEST_WINDOWS}
    physics = {w: score(t, *dead_reckoning_offsets(t)) for w, t in clean_tests.items()}

    rows = []
    for condition in conditions:
        t0 = time.perf_counter()
        # Everything that arrived in W1-W5 for this condition, turned into training examples.
        stream = make_dataset(load_pings(source, condition, windows=[1, 2, 3, 4, 5], seed=seed))
        defects = defect_share_by_window(condition).set_index("window_id") if source == "real" else None

        for test_w in TEST_WINDOWS:
            test = clean_tests[test_w]
            cutoff = test["timestamp"].min()
            known_in_time = stream["timestamp"] + pd.Timedelta(seconds=HORIZON_S) <= cutoff
            train = stream[(stream["window_id"] < test_w) & known_in_time].reset_index(drop=True)

            models = train_models(train, seed)
            start = time.perf_counter()
            north, east = predict_offsets(models, test)
            predict_ms = (time.perf_counter() - start) * 1000

            ai = score(test, north, east)
            row = {
                "condition": condition,
                "mitigated": False,
                "train_windows": f"W1-W{test_w - 1}",
                "window_id": test_w,  # the window being predicted
                "n_train_rows": len(train),
                "n_train_voyages": int(train["voyage_id"].nunique()),
                "n_test_rows": len(test),
                "n_test_voyages": int(test["voyage_id"].nunique()),
                "displacement_mae_m": ai["displacement_mae_m"],
                "tail_95_error_m": ai["tail_95_error_m"],
                "by_vessel_group": ai["by_vessel_group"],
                "physics_mae_m": physics[test_w]["displacement_mae_m"],
                "physics_tail_95_m": physics[test_w]["tail_95_error_m"],
                "data_discarded_pct": 0.0,  # no countermeasure: nothing is filtered
                "decisions_deferred_pct": 0.0,  # no countermeasure: nothing goes to a human
                "predict_total_ms": round(predict_ms, 2),
                "predict_ms_per_row": round(predict_ms / len(test), 5),
            }
            if defects is not None:
                # REPORTING ONLY (never given to the model): how corrupted the training buffer was.
                buf = defects.loc[defects.index < test_w]
                row["train_buffer_corrupted_pct"] = round(100.0 * buf["corrupted_rows"].sum() / buf["rows"].sum(), 2)
                row["train_buffer_tier2_corrupted_pct"] = round(
                    100.0 * buf["corrupted_rows"].sum() / buf["tier2_rows"].sum(), 2)
            rows.append(row)
            print(f"  {condition:12} train W1-W{test_w - 1} ({len(train):>7,} rows) -> test W{test_w}: "
                  f"{ai['displacement_mae_m']:7.0f} m avg, {ai['tail_95_error_m']:7.0f} m 95th")

            if test_w == TEST_WINDOWS[-1]:  # keep the final W1-W5 model for the app
                save_models(models, f"rolling_unmitigated_{condition.lower()}_{source}", info={
                    "condition": condition, "train_windows": "W1-W5", "source": source, "seed": seed})
        print(f"  {condition} done in {time.perf_counter() - t0:.0f} s")

    return {
        "run": "rolling_retrain_unmitigated",
        "data_source": source,
        "seed": seed,
        "conditions": conditions,
        "design": {
            "steps": "train W1..Wk (everything that arrived, corrupted rows included) -> test W(k+1)",
            "test_data": "clean NOMINAL inputs and true positions, identical for all conditions",
            "poisoning_stage": "training-time (retraining buffer); test-time inputs are clean",
            "no_future_rule": "train rows need their 15-min answer before the test window's first ping",
            "window_rule": "voyage belongs to the window of its first ping",
        },
        "lgbm_params": lgbm_params(seed),
        "rows": rows,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def print_tables(result: dict) -> None:
    df = pd.DataFrame(result["rows"])
    conds = [c for c in CONDITIONS if c in set(df["condition"])]
    for title, col, phys in (("Average error (m)", "displacement_mae_m", "physics_mae_m"),
                             ("95th-percentile error (m)", "tail_95_error_m", "physics_tail_95_m")):
        t = df.pivot(index="window_id", columns="condition", values=col)[conds]
        t["PHYSICS"] = df.groupby("window_id")[phys].first()
        t.index = [f"test W{w} (trained W1-W{w - 1})" for w in t.index]
        print(f"\n{title}, no countermeasure:\n{t.round(0).astype(int).to_string()}")

    cargo = df.assign(cargo=df["by_vessel_group"].map(lambda g: g.get("cargo", {}).get("displacement_mae_m", np.nan)))
    t = cargo.pivot(index="window_id", columns="condition", values="cargo")[conds]
    t.index = [f"test W{w}" for w in t.index]
    print(f"\nCargo ships only, average error (m):\n{t.round(0).to_string()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly retraining without the countermeasure.")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", choices=["real"], default="real")
    args = parser.parse_args()

    print(f"Rolling retraining, no countermeasure | {args.source} data | seed {args.seed}")
    result = run_rolling(args.conditions, seed=args.seed, source=args.source)
    out = REPO_ROOT / "results" / f"rolling_unmitigated_{args.source}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print_tables(result)
    print(f"\nSaved: {out.relative_to(REPO_ROOT)} (+ final W1-W5 models in models/)")


if __name__ == "__main__":
    main()
