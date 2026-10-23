"""Grade the trust gate like a detector (case requirement: false positives / negatives).

Run from the repository root:
    uv run python -m backend.ml.grade_gate

The gate makes its decisions WITHOUT the hidden defect flag. Only afterwards do we
reveal the flag and count, for TIER2_FAST pings in each week:

    caught       (true positive)   corrupted ping, gate flagged it
    missed       (false negative)  corrupted ping, gate kept it
    false alarm  (false positive)  good ping, gate flagged it (good data thrown away)
    correct keep (true negative)   good ping, gate kept it

"Positive" = a corrupted ping. Rates reported:
    catch rate        = caught / all corrupted          (recall)
    precision         = caught / all flagged            (how often a flag is right)
    false alarm rate  = false alarms / all good pings
    major catch rate  = catch rate for MAJOR corruptions only: position moved > 200 m
                        or speed changed > 3 kn vs. the clean value. Tiny corruptions
                        barely matter, so this shows whether the gate catches the harmful ones.

Limits are learned from each condition's own week W1 (clean in every condition).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.ml.data import CONDITIONS, REPO_ROOT, load_audit_labels, load_pings
from backend.ml.gate import LIMIT_PERCENTILE, compare_to_anchor, learn_limits, score_trust
from backend.ml.metrics import haversine_m

GROUP_NAMES = {0: "cargo", 1: "tanker", 2: "tug", 3: "other"}
MAJOR_POSITION_M = 200.0
MAJOR_SPEED_KN = 3.0


def confusion(flagged: np.ndarray, corrupted: np.ndarray) -> dict:
    """Confusion counts and rates. Positive class = corrupted ping."""
    tp = int(np.sum(flagged & corrupted))
    fp = int(np.sum(flagged & ~corrupted))
    fn = int(np.sum(~flagged & corrupted))
    tn = int(np.sum(~flagged & ~corrupted))
    rate = lambda a, b: round(100.0 * a / b, 2) if b else None
    return {
        "caught_tp": tp, "false_alarm_fp": fp, "missed_fn": fn, "correct_keep_tn": tn,
        "catch_rate_pct": rate(tp, tp + fn),
        "precision_pct": rate(tp, tp + fp),
        "false_alarm_rate_pct": rate(fp, fp + tn),
        "flagged_pct": rate(tp + fp, tp + fp + fn + tn),
    }


def grade_condition(condition: str) -> dict:
    pings = load_pings("real", condition)
    checked = compare_to_anchor(pings)
    limits = learn_limits(checked[checked["window_id"] == 1])
    scored = score_trust(checked, limits)

    # Reveal the answers only now, after the gate has decided.
    labels = load_audit_labels(condition)
    graded = scored.merge(labels, on=["voyage_id", "timestamp"], how="left", validate="one_to_one")
    assert graded["is_injected_defect"].notna().all(), "every ping must have an audit label"
    fast = graded[graded["source_tier"] == "TIER2_FAST"].copy()
    fast["corrupted"] = fast["is_injected_defect"].astype(bool)
    moved_m = haversine_m(fast["lat"], fast["lon"], fast["clean_lat"], fast["clean_lon"])
    fast["major"] = fast["corrupted"] & ((moved_m > MAJOR_POSITION_M)
                                         | (np.abs(fast["sog"] - fast["clean_sog"]) > MAJOR_SPEED_KN))

    by_window = {}
    for w, d in fast.groupby("window_id"):
        flagged, corrupted, major = d["suspect"].to_numpy(), d["corrupted"].to_numpy(), d["major"].to_numpy()
        stats = confusion(flagged, corrupted)
        stats["corrupted_pct"] = round(100.0 * corrupted.mean(), 2)
        stats["major_corrupted_pct"] = round(100.0 * major.mean(), 2)
        stats["major_catch_rate_pct"] = round(100.0 * flagged[major].mean(), 2) if major.any() else None
        stats["unverifiable_pings"] = int((~d["verifiable"]).sum())
        by_window[f"W{int(w)}"] = stats

    good = fast[~fast["corrupted"]]
    false_alarms_by_group = {
        GROUP_NAMES[int(g)]: round(float(100.0 * d["suspect"].mean()), 2) for g, d in good.groupby("vessel_group")
    }
    major = fast[fast["major"]]
    major_caught_by_group = {
        GROUP_NAMES[int(g)]: {"major_corrupted_pings": len(d), "caught_pct": round(float(100.0 * d["suspect"].mean()), 2)}
        for g, d in major.groupby("vessel_group")
    }
    return {
        "limits_learned_from_W1": limits,
        "by_window": by_window,
        "overall": confusion(fast["suspect"].to_numpy(), fast["corrupted"].to_numpy()),
        "false_alarm_rate_pct_by_ship_type": false_alarms_by_group,
        "major_catch_rate_by_ship_type": major_caught_by_group,
    }


def main() -> None:
    print("Grading the trust gate (TIER2_FAST pings; the gate never saw the defect flag)\n")
    results = {}
    for condition in CONDITIONS:
        r = grade_condition(condition)
        results[condition] = r
        t = pd.DataFrame(r["by_window"]).T[
            ["corrupted_pct", "catch_rate_pct", "major_corrupted_pct", "major_catch_rate_pct",
             "false_alarm_rate_pct", "precision_pct", "flagged_pct"]]
        t.columns = ["corrupted %", "caught %", "major %", "major caught %", "false alarm %", "precision %",
                     "thrown out %"]
        print(f"{condition}\n{t.to_string(na_rep='-')}")
        print(f"  false alarm % by ship type:   {r['false_alarm_rate_pct_by_ship_type']}")
        major = {g: v["caught_pct"] for g, v in r["major_catch_rate_by_ship_type"].items()}
        print(f"  major caught % by ship type:  {major or '-'}\n")

    out = REPO_ROOT / "results" / "gate_grading_real.json"
    out.write_text(json.dumps({
        "run": "trust_gate_grading",
        "positive_class": "corrupted TIER2_FAST ping (is_injected_defect), revealed only after gating",
        "limit_rule": f"{LIMIT_PERCENTILE}th percentile of each check on the condition's clean week W1, per ship type",
        "major_corruption": f"position moved > {MAJOR_POSITION_M} m or speed changed > {MAJOR_SPEED_KN} kn",
        "conditions": results,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2))
    print(f"Saved: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
