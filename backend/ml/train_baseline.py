"""Reference baseline: train LightGBM to predict where a ship will be in 15 minutes.

Run from the repository root:
    uv run python -m backend.ml.train_baseline                  # dummy data, window W1, seed 42
    uv run python -m backend.ml.train_baseline --source real    # Person 1's real data
    uv run python -m backend.ml.train_baseline --seed 7         # repeat with a different seed

What it does:
 1. Load clean (NOMINAL) pings for one window: dummy data, or Person 1's database (data.py).
 2. Build features and 15-minute targets (features.py).
 3. Split by whole voyage, in time order: earliest ~80% of voyages train, latest ~20% test.
 4. Start from a physics guess ("dead reckoning": the ship keeps its current speed and
    heading). Train two LightGBM models to learn the CORRECTION to that guess:
    one for meters north, one for meters east. Final prediction = physics + correction.
    This lets the AI focus on what physics misses (e.g. turning at a channel bend).
 5. Score in meters against the true position, and also score physics alone, so we can
    show the AI adds value (if it doesn't, we report that honestly).
 6. Save the numbers to results/baseline_w<window>_<source>.json and the trained
    models to models/baseline_w<window>_<source>.joblib.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.ml.data import SOURCES, load_pings
from backend.ml.features import (
    FEATURE_COLUMNS,
    HORIZON_S,
    TARGET_COLUMNS,
    dead_reckoning_offsets,
    make_dataset,
)
from backend.ml.metrics import haversine_m, offset_m_to_latlon, summarize_errors
from backend.ml.model_store import save_models

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def lgbm_params(seed: int) -> dict:
    """Model settings. Fixed seed + deterministic mode -> same results on every run."""
    return {
        "n_estimators": 300,  # number of small decision trees added one after another
        "learning_rate": 0.05,  # how much each tree is allowed to correct the previous ones
        "num_leaves": 31,  # how detailed each tree can be
        "min_child_samples": 20,  # each tree leaf must cover at least 20 rows (limits overfitting)
        "objective": "l1",  # learn the typical (median) correction, so rare wild cases don't dominate
        "random_state": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }


def split_by_voyage_in_time(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep each voyage whole; the latest-starting voyages become the test set.

    To be strict about the future, training only keeps rows whose 15-minute
    answer was known before the first test voyage began.
    """
    starts = df.groupby("voyage_id")["timestamp"].min().sort_values()
    if len(starts) < 2:
        raise ValueError("Need at least 2 voyages to split into train and test.")
    n_test = max(1, round(len(starts) * test_frac))
    test_ids = starts.index[-n_test:]
    cutoff = starts.iloc[-n_test]  # when the first test voyage starts

    is_test = df["voyage_id"].isin(test_ids)
    answer_known_in_time = df["timestamp"] + pd.Timedelta(seconds=HORIZON_S) <= cutoff
    train = df[~is_test & answer_known_in_time].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    return train, test


def train_models(train: pd.DataFrame, seed: int) -> dict[str, LGBMRegressor]:
    """One model per direction, each learning the correction to the physics guess."""
    physics = dict(zip(("north", "east"), dead_reckoning_offsets(train)))
    models = {}
    for direction, target in zip(("north", "east"), TARGET_COLUMNS):
        correction = train[target].to_numpy() - physics[direction]
        model = LGBMRegressor(**lgbm_params(seed))
        model.fit(train[FEATURE_COLUMNS], correction)
        models[direction] = model
    return models


def predict_offsets(models: dict[str, LGBMRegressor], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predicted meters (north, east) in 15 minutes = physics guess + learned correction."""
    phys_north, phys_east = dead_reckoning_offsets(df)
    north = phys_north + models["north"].predict(df[FEATURE_COLUMNS])
    east = phys_east + models["east"].predict(df[FEATURE_COLUMNS])
    return north, east


GROUP_NAMES = {0: "cargo", 1: "tanker", 2: "tug", 3: "other"}


def errors_m(df: pd.DataFrame, north_m: np.ndarray, east_m: np.ndarray) -> np.ndarray:
    """Convert predicted movement into a position and measure each miss in meters."""
    pred_lat, pred_lon = offset_m_to_latlon(df["lat"], df["lon"], north_m, east_m)
    return haversine_m(pred_lat, pred_lon, df["target_lat"], df["target_lon"])


def score(df: pd.DataFrame, north_m: np.ndarray, east_m: np.ndarray) -> dict:
    """Overall error numbers, plus the same numbers for each vessel group."""
    errs = errors_m(df, north_m, east_m)
    result = summarize_errors(errs)
    groups = df["vessel_group"].to_numpy()
    result["by_vessel_group"] = {
        GROUP_NAMES[int(g)]: summarize_errors(errs[groups == g]) for g in np.unique(groups)
    }
    return result


def run_baseline(window: int = 1, seed: int = 42, source: str = "dummy", save: bool = False) -> dict:
    pings = load_pings(source=source, condition="NOMINAL", windows=[window], seed=seed)
    dataset = make_dataset(pings)
    train, test = split_by_voyage_in_time(dataset)

    models = train_models(train, seed)
    start = time.perf_counter()
    north_pred, east_pred = predict_offsets(models, test)
    predict_ms = (time.perf_counter() - start) * 1000

    result = {
        "run": "reference_baseline",
        "data_source": source,
        "is_synthetic": source == "dummy",
        "window_id": window,
        "seed": seed,
        "horizon_minutes": HORIZON_S / 60,
        "split_rule": "whole voyages; latest-starting 20% are test; train rows need t+15 before test start",
        "n_train_voyages": int(train["voyage_id"].nunique()),
        "n_test_voyages": int(test["voyage_id"].nunique()),
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "features": FEATURE_COLUMNS,
        "model": "dead reckoning + LightGBM correction (one regressor each for north and east meters)",
        "lgbm_params": lgbm_params(seed),
        "lightgbm": score(test, north_pred, east_pred),
        "dead_reckoning": score(test, *dead_reckoning_offsets(test)),
        "predict_total_ms": round(predict_ms, 2),
        "predict_ms_per_row": round(predict_ms / len(test), 5),
        "data_notes": pings.attrs.get("notes", {}),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if save:
        path = save_models(models, f"baseline_w{window}_{source}", info={
            k: result[k] for k in ("run", "data_source", "window_id", "seed", "split_rule", "lgbm_params")
        })
        result["saved_model"] = str(path.relative_to(REPO_ROOT))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and score the 15-minute position baseline.")
    parser.add_argument("--window", type=int, default=1, help="which week to use, 1-6 (default 1)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--source", choices=SOURCES, default="dummy", help="dummy (default) or real")
    args = parser.parse_args()

    r = run_baseline(window=args.window, seed=args.seed, source=args.source, save=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"baseline_w{r['window_id']}_{r['data_source']}.json"
    out_path.write_text(json.dumps(r, indent=2))

    label = "SYNTHETIC DUMMY DATA" if r["is_synthetic"] else r["data_source"]
    print(f"Reference baseline | window W{r['window_id']} | {label} | seed {r['seed']}")
    print(f"Train: {r['n_train_voyages']} voyages, {r['n_train_rows']:,} rows")
    print(f"Test:  {r['n_test_voyages']} voyages, {r['n_test_rows']:,} rows\n")
    print(f"{'':24}{'AI model':>22}{'Physics only':>22}")
    print(f"{'':24}{'avg (m)':>11}{'95th (m)':>11}{'avg (m)':>11}{'95th (m)':>11}")
    ai, phys = r["lightgbm"], r["dead_reckoning"]
    rows = [("ALL SHIPS", ai, phys)] + [
        (f"  {g} ({ai['by_vessel_group'][g]['n_predictions']:,} rows)", ai["by_vessel_group"][g], phys["by_vessel_group"][g])
        for g in ai["by_vessel_group"]
    ]
    for name, a, p in rows:
        print(f"{name:24}{a['displacement_mae_m']:>11.0f}{a['tail_95_error_m']:>11.0f}"
              f"{p['displacement_mae_m']:>11.0f}{p['tail_95_error_m']:>11.0f}")
    print(f"\nPrediction time: {r['predict_total_ms']:.1f} ms for {r['n_test_rows']:,} rows")
    print(f"Saved: {out_path.relative_to(REPO_ROOT)} and {r['saved_model']}")


if __name__ == "__main__":
    main()
