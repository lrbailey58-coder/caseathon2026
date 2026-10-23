"""Step 6: one summary of all 8 experiments (4 conditions x gate off/on) for the presentation.

Run from the repository root, AFTER rolling.py and grade_gate.py:
    uv run python -m backend.ml.summary

What it produces:
  1. The 8-run table for the final, previously unseen week (W6): error, data thrown
     out, predictions handed to a human, gate catch rate, and damage undone.
  2. An uncertainty check (the case asks for one). We have one run, so we ask: "if we
     had tested on a different set of voyages, would the gaps still be there?" We
     re-sample the W6 test voyages with replacement 1,000 times (a bootstrap) and
     report a 95% range for each number. Voyages are re-sampled whole, because pings
     from one voyage are not independent. If a 95% range for a gap includes 0, that
     gap could be luck.
  3. Saves results/experiment_summary_real.json (machine-readable) and
     results/experiment_summary_real.md (a table to paste into slides).

The W6 numbers are recomputed from the SAVED W1-W5 models, and the script checks
they match rolling.py's numbers exactly (a consistency check).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.ml.data import CONDITIONS, REPO_ROOT, load_pings
from backend.ml.features import make_dataset
from backend.ml.model_store import MODELS_DIR, load_models
from backend.ml.rolling import model_name
from backend.ml.train_baseline import errors_m, predict_offsets

RESULTS = REPO_ROOT / "results"
FINAL_WINDOW = 6
N_BOOT = 1000
BOOT_SEED = 0
LABEL = {"NOMINAL": "Clean", "STALE": "Stale", "LOW_QUALITY": "Noisy", "COMBINED": "Both"}


def _read(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run rolling.py and grade_gate.py first.")
    return json.loads(path.read_text())


def w6_errors() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-prediction W6 errors for all 8 runs, using the saved W1-W5 models."""
    test = make_dataset(load_pings("real", "NOMINAL", windows=[FINAL_WINDOW]))
    errs = {}
    for cond in CONDITIONS:
        for tag in ("unmitigated", "gated"):
            bundle = load_models(MODELS_DIR / f"{model_name(tag, cond, 'real', FINAL_WINDOW)}.joblib")
            errs[f"{cond}|{tag == 'gated'}"] = errors_m(test, *predict_offsets(bundle["models"], test))
    return test, errs


def bootstrap(test: pd.DataFrame, errs: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    """95% ranges for each run's average error and for the key gaps, re-sampling whole voyages."""
    voyages, idx = np.unique(test["voyage_id"].to_numpy()[mask], return_inverse=True)
    counts = np.bincount(idx)
    sums = {k: np.bincount(idx, weights=e[mask]) for k, e in errs.items()}  # error total per voyage
    rng = np.random.default_rng(BOOT_SEED)
    picks = rng.integers(0, len(voyages), size=(N_BOOT, len(voyages)))  # same voyages for every run (paired)
    n = counts[picks].sum(axis=1)
    mae = {k: s[picks].sum(axis=1) / n for k, s in sums.items()}
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 1), round(float(np.percentile(x, 97.5)), 1)]

    out = {"voyages": int(len(voyages)), "runs": {k: ci(v) for k, v in mae.items()}, "gaps": {}}
    clean = mae["NOMINAL|False"]
    for cond in CONDITIONS[1:]:
        off, on = mae[f"{cond}|False"], mae[f"{cond}|True"]
        out["gaps"][cond] = {
            "damage_m (gate off - clean)": ci(off - clean),
            "gate_gain_m (gate off - gate on)": ci(off - on),
            "left_over_m (gate on - clean)": ci(on - clean),
        }
    out["gaps"]["NOMINAL"] = {"gate_cost_on_clean_m (gate on - gate off)": ci(mae["NOMINAL|True"] - clean)}
    return out


def gate_catch(grading: dict, cond: str) -> float | None:
    """Share of MAJOR corruptions the gate caught, over all weeks and ship types."""
    groups = grading["conditions"][cond]["major_catch_rate_by_ship_type"]
    total = sum(g["major_corrupted_pings"] for g in groups.values())
    if total == 0:
        return None
    return round(sum(g["caught_pct"] * g["major_corrupted_pings"] for g in groups.values()) / total, 1)


def build() -> dict:
    rolling = _read("rolling_real.json")
    grading = _read("gate_grading_real.json")
    rows = pd.DataFrame(rolling["rows"])
    final = rows[rows["window_id"] == FINAL_WINDOW].set_index(["condition", "mitigated"])

    test, errs = w6_errors()
    # Consistency check: saved models must reproduce rolling.py's W6 numbers exactly.
    for key, e in errs.items():
        cond, mitigated = key.split("|")
        expected = final.loc[(cond, mitigated == "True"), "displacement_mae_m"]
        assert abs(e.mean() - expected) < 1e-6, f"{key}: saved model gives {e.mean():.3f}, rolling.py {expected:.3f}"

    is_cargo = (test["vessel_group"] == 0).to_numpy()
    boot_all = bootstrap(test, errs, np.ones(len(test), bool))
    boot_cargo = bootstrap(test, errs, is_cargo)

    clean_all = final.loc[("NOMINAL", False), "displacement_mae_m"]
    clean_cargo = errs["NOMINAL|False"][is_cargo].mean()
    table = []
    for cond in CONDITIONS:
        for mitigated in (False, True):
            r = final.loc[(cond, mitigated)]
            cargo = float(errs[f"{cond}|{mitigated}"][is_cargo].mean())
            row = {
                "condition": cond,
                "countermeasure": "gate + human hand-off" if mitigated else "none",
                "avg_error_m": round(float(r["displacement_mae_m"]), 1),
                "avg_error_95pct_range_m": boot_all["runs"][f"{cond}|{mitigated}"],
                "tail_95_error_m": round(float(r["tail_95_error_m"]), 1),
                "cargo_avg_error_m": round(cargo, 1),
                "cargo_avg_error_95pct_range_m": boot_cargo["runs"][f"{cond}|{mitigated}"],
                "data_discarded_pct": float(r["data_discarded_pct"]),
                "handed_to_human_pct": float(r["decisions_deferred_pct"]),
                "gate_major_catch_pct": gate_catch(grading, cond) if (mitigated and cond != "NOMINAL") else None,
                "physics_avg_error_m": round(float(r["physics_mae_m"]), 1),
            }
            if mitigated and cond != "NOMINAL":
                off_all = final.loc[(cond, False), "displacement_mae_m"]
                off_cargo = errs[f"{cond}|False"][is_cargo].mean()
                row["damage_undone_pct_all"] = round(100 * (off_all - r["displacement_mae_m"]) / (off_all - clean_all), 1)
                row["damage_undone_pct_cargo"] = round(100 * (off_cargo - cargo) / (off_cargo - clean_cargo), 1)
            table.append(row)

    return {
        "run": "experiment_summary",
        "final_window": f"W{FINAL_WINDOW} (previously unseen; models trained on W1-W5)",
        "test_cases": {"predictions": len(test), "voyages": boot_all["voyages"], "cargo_voyages": boot_cargo["voyages"]},
        "eight_runs": table,
        "uncertainty_check": {
            "method": f"bootstrap over whole test voyages, {N_BOOT} resamples, paired across runs, seed {BOOT_SEED}",
            "reading": "95% range of each number; a gap whose range includes 0 could be luck",
            "all_ships": boot_all,
            "cargo": boot_cargo,
        },
        "notes": [
            "Error numbers include every prediction, also the ones handed to a human.",
            "Single training run; LightGBM has no randomness here, so variability comes from which voyages were tested.",
            f"Tanker numbers are omitted: W6 has only {int(test.loc[test['vessel_group'] == 1, 'voyage_id'].nunique())} "
            "tanker voyages.",
        ],
        "sources": ["results/rolling_real.json", "results/gate_grading_real.json", "models/rolling_*_real_w6.joblib"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def to_markdown(s: dict) -> str:
    fmt = lambda v, unit="": "-" if v is None else f"{v:,.0f}{unit}"
    lines = [
        f"## Results: final unseen week {s['final_window']}",
        "",
        "| Data | Countermeasure | Avg error (all) | Avg error (cargo) | 95th pct (all) | Data thrown out "
        "| Handed to human | Major corruptions caught | Damage undone (cargo) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in s["eight_runs"]:
        lines.append(
            f"| {LABEL[r['condition']]} | {r['countermeasure']} | {fmt(r['avg_error_m'], ' m')} "
            f"| {fmt(r['cargo_avg_error_m'], ' m')} | {fmt(r['tail_95_error_m'], ' m')} "
            f"| {r['data_discarded_pct']:.1f}% | {r['handed_to_human_pct']:.1f}% "
            f"| {fmt(r['gate_major_catch_pct'], '%')} | {fmt(r.get('damage_undone_pct_cargo'), '%')} |")
    lines += ["", f"Physics-only baseline: {s['eight_runs'][0]['physics_avg_error_m']:,.0f} m average error.", "",
              "### Is it luck? 95% ranges for the gaps (cargo ships, meters)", "",
              "| Data | Damage from poisoning | Gained back by the gate | Left over vs clean |", "|---|---|---|---|"]
    gaps = s["uncertainty_check"]["cargo"]["gaps"]
    for cond in CONDITIONS[1:]:
        g = gaps[cond]
        rng = lambda k: f"{g[k][0]:+.0f} to {g[k][1]:+.0f}"
        lines.append(f"| {LABEL[cond]} | {rng('damage_m (gate off - clean)')} | {rng('gate_gain_m (gate off - gate on)')} "
                     f"| {rng('left_over_m (gate on - clean)')} |")
    cost = gaps["NOMINAL"]["gate_cost_on_clean_m (gate on - gate off)"]
    lines += ["", f"Cost of the gate on clean data (cargo): {cost[0]:+.0f} to {cost[1]:+.0f} m.", ""]
    lines += [f"- {n}" for n in s["notes"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    s = build()
    (RESULTS / "experiment_summary_real.json").write_text(json.dumps(s, indent=2))
    md = to_markdown(s)
    (RESULTS / "experiment_summary_real.md").write_text(md)
    print(md)
    print("Saved: results/experiment_summary_real.json and results/experiment_summary_real.md")


if __name__ == "__main__":
    main()
