# ML Pipeline: Prediction Model, Trust Gate & Human Hand-Off

Owner: Jay Lavender (ML & Countermeasure Engineer). This folder holds everything that trains,
poisons, protects and evaluates the AI. The website only displays the results it produces.

## 1. What the AI does

**Question:** where will this ship be 15 minutes from now?

- **Physics guess (the normal method):** assume the ship keeps its current speed and heading
  ("dead reckoning").
- **The AI:** two LightGBM models (gradient-boosted decision trees) learn a **correction** to that
  physics guess, in meters north and meters east, from past voyages: for example, ships turning at
  a channel bend or slowing near port. Prediction = physics guess + learned correction.
- **How sure it is:** four more LightGBM models (quantile regression, 5th and 95th percentile)
  predict a 90% range. If that range is wider than 1 nautical mile, the prediction is handed to
  the Watch Officer instead of being trusted automatically.

The AI component is LightGBM, trained here on real AIS data. No large language model makes any
prediction or decision in this system.

## 2. How to run (from the repository root)

Run in this order. Needs `backend/maritime_case.duckdb` (built by Person 1's
`backend/database/build_database.py`). About 15 minutes in total; `rolling` is the slow step.

| # | Command | What it does | Output |
|---|---|---|---|
| 1 | `uv run python -m backend.ml.check_data` | 22 data checks before training (PASS / WARN / FAIL); exits with code 1 on any FAIL | `results/data_check_real.json` |
| 2 | `uv run python -m backend.ml.train_baseline --source real` | Reference model on clean week W1, compared with physics | `results/baseline_w1_real.json`, `models/baseline_w1_real.joblib` |
| 3 | `uv run python -m backend.ml.rolling` | Weekly retraining W1..W5 for 4 data conditions x gate off/on, tested on the next week | `results/rolling_real.json`, `models/rolling_*.joblib` |
| 4 | `uv run python -m backend.ml.grade_gate` | Grades the trust gate like a detector (catches, misses, false alarms) | `results/gate_grading_real.json` |
| 5 | `uv run python -m backend.ml.summary` | 8-run table for unseen week W6 plus a bootstrap "is it luck?" check | `results/experiment_summary_real.{json,md}` |
| 6 | `uv run python -m backend.ml.backfire` | Failure cases and tradeoffs: clean-data cost, sharp turns, new corruptions | `results/backfire_checks_real.json` |
| 7 | `uv run python -m backend.ml.export_dashboard` | Exports results for the website (the backend loads this file on start) | `results/dashboard_export_real.json` |

Step 1 must show 0 FAIL before going on. After step 7, commit `results/` so the hosted site shows the
new numbers. `models/` is git-ignored: rerun the scripts to rebuild it.

Without the database, `uv run python -m backend.ml.train_baseline` runs on synthetic dummy data
(`dummy_data.py`) to test the code path.

## 3. Data rules (applied in `data.py`, identically in every condition)

Source: NOAA MarineCadastre AIS, 2025-09-01 to 2025-10-12, Norfolk / Chesapeake Bay box
(36.8-37.3 N, 76.6-75.8 W), tugs/towing (AIS types 31, 32, 52), cargo (70-79), tankers (80-89),
speed > 0.5 kn: 671,747 pings, 9,680 voyages (see `backend/database/manifest.json`).

| Rule | Why | Rows affected |
|---|---|---|
| A voyage belongs to the window of its first ping | keeps each voyage whole (5 voyages had pings in two windows) | 5 voyages |
| Drop rows with missing lat/lon/speed/heading | the physics guess needs them | 120 |
| Drop pings whose original (clean) speed is > 40 kn | source glitch (a tug reporting 46.9 kn while moving 2.4 kn) | 1 |
| Drop voyages with 2+ jumps implying > 50 kn | two ships sharing one ID (e.g. 200000000, positions 40 km apart) | 4 voyages |
| Drop both copies of duplicate timestamps in a voyage | no rule says which copy is right; keeping one made runs non-repeatable | 2,049 |

These are judged on the **clean** reference data, so corruption we injected on purpose is never
removed here (that is the gate's job). The model never sees `is_injected_defect`: only
`check_data.py` and `grade_gate.py` read it, for auditing after decisions are made.

## 4. Experiment design (`rolling.py`)

- **Windows:** 6 roughly weekly windows (W1 clean in every condition; W6 = final unseen test).
- **Steps:** retrain on everything that arrived in W1..W(k-1), test on Wk, for k = 2..6.
- **Training buffer:** both feeds, earlier windows, timestamps before the moment of retraining. The
  slow anchor feed arrives 45 minutes late, so newer anchor pings are not in the buffer yet. A
  training row only counts if its 15-minute answer was known before retraining (no future leakage).
- **Gate off ("naive retraining"):** train on the whole buffer, corrupted rows included.
- **Gate on:** the trust gate drops suspect fast-feed pings first; uncertain predictions are handed off.
- **Testing:** always clean data (NOMINAL inputs, true positions), so all 8 runs are scored on the
  same cases and any difference comes from training alone. This measures **training-time** poisoning.
- **Error metric:** distance in meters between the predicted and true position 15 minutes later:
  average (MAE) and 95th percentile. Every prediction counts, including those handed to a human.
- **Settings:** LightGBM, 300 trees, learning rate 0.05, 31 leaves, `objective="l1"`, seed 42,
  deterministic mode. Features: lat, lon, speed, heading (sin/cos), velocity north/east, time since
  the last ping, speed change and turn rate per minute, vessel group.

## 5. The trust gate (`gate.py`)

For every fast-feed ping, the gate compares it with where the slow, trusted feed says the ship was
at that moment (straight-line interpolation between anchor pings about 6 minutes apart):
**position** (meters off), **speed** (knots off), **heading** (degrees off; only above 3 kn).

- **Limits** = 99th percentile of each check on clean week W1's fast feed, **per ship type**
  (cargo 326 m / 1.7 kn / 25 deg; tankers 507 m / 4.0 kn / 43 deg; tugs 616 m / 5.4 kn / 130 deg;
  a type with < 500 reference pings would use the all-ship limits). Learned only from W1, never
  tuned on any corruption (see `results/gate_grading_real.json`).
- **Trust score** = min(1, limit / value) over the checks; below 1 = suspect = dropped before
  retraining. Anchor pings are always kept; pings with no anchor nearby are kept (unverifiable).

## 6. Human hand-off (`uncertainty.py`)

Uncertainty radius = half-diagonal of the 90% range box (5th to 95th percentile, north and east).
Radius > **1,852 m (1 nautical mile)**, a common closest-point-of-approach alarm distance fixed in
advance, means the prediction is flagged "needs Watch Officer review."

## 7. Results (unseen week W6, models trained on W1-W5)

1,167 test voyages, 88,812 predictions. Physics alone: 995 m. Full table: `results/experiment_summary_real.md`.

| Data | Gate | Avg error (all) | Avg error (cargo) | Data dropped | Handed to human | Damage undone (all / cargo) |
|---|---|---|---|---|---|---|
| Clean | off | 667 m | 543 m | 0% | 0% | - |
| Clean | on | 671 m | 547 m | 1.9% | 18.8% | (cost: +0.5% / +0.8%) |
| Stale | off / on | 690 / 669 m | 582 / 546 m | 22.6% | 18.8% | 91% / 92% |
| Noisy | off / on | 683 / 667 m | 566 / 553 m | 16.4% | 19.6% | 99% / 56% |
| Both | off / on | 707 / 665 m | 608 / 546 m | 30.9% | 18.9% | 105% / 96% |

- **The AI beats physics:** 604 vs 920 m on W1 (cargo: 539 vs 1,024 m).
- **Poisoning hurts, and grows over time:** for cargo, "both" goes from equal at W2 to +65 m at W6.
- **Is it luck?** Re-sampling the 1,167 test voyages 1,000 times (paired bootstrap): the damage
  (all ships, both: +35 to +47 m) and the gate's gain (+37 to +48 m) never include 0. What is left
  after the gate is within luck for all ships and "both" (-3.6 to -0.2 m), but real for noisy cargo
  (+7 to +13 m).
- **Gate as a detector:** catches 95% (stale), 50% (noisy) and 78% (both) of major corruptions
  (moved > 200 m or speed changed > 3 kn); 99.8% for cargo with stale data. False alarms on clean
  data: 2.4%.
- **Hand-offs are the right ones:** handed-off predictions average about 1,500 m off versus about
  470 m for the ones the AI keeps; about two-thirds of the worst 10% are handed off. The 90% range
  contains the truth 88-91% of the time.

## 8. Limits, failure cases and next step

- **Real sharp turns are thrown out as noise.** The gate's straight line between anchor pings
  misses curves: on clean data it wrongly flags 46% of cargo reports in sharp turns (20-60 deg/min)
  and 88% in very sharp turns, versus 1.1% when steady. The map's "Tug Turn" view shows a real
  example.
- **Noisy tugs:** only 40% of major noise corruptions on tugs are caught (their normal limits are wide).
- **Steady GPS offsets are missed:** a 300 m fixed shift was caught 8% of the time (it barely hurt
  the model, because a constant shift cancels out in the movement it learns).
- **No coastline knowledge:** in narrow rivers, 15-minute forecasts can cut corners onto land.
- **Data scope:** 81% of voyages are tugs; tankers are too few (9 in W6) to report separately. The
  injected corruption is simulated, not observed feed failures. One training run per setting
  (LightGBM is deterministic here); uncertainty comes from the bootstrap over test voyages.
- **Bounded next step:** a turn-aware gate that compares against a curved path (anchor positions
  plus their headings). Owner: ML lead. Checkpoint before any pilot: sharp-turn false alarms on
  clean data below 10%, with no loss of poisoning recovery at W6.

## 9. Reproducibility

Same data + same code = same numbers on the same computer (seed 42, deterministic LightGBM; the
summary script refuses to run if saved models do not reproduce `rolling.py` exactly). Different
computers can differ slightly (well under 1%). Compute times (about 6 ms per prediction, 14 ms with
the range models, measured on our laptop) are re-measured on every export and vary with machine load.

## 10. AI tool and prompt disclosure (ML part)

- **AI inside the system:** LightGBM models trained by this code (section 1). No generative AI at
  run time.
- **AI tools used to build it:** Anthropic Claude (Cowork, in the Claude desktop app) helped write and
  test this pipeline's code, find data problems (duplicate timestamps, the speed glitch, shared
  ship IDs) and explain results; Google Gemini was used for the initial architecture plan
  (`Architecture.md`). Typical prompts: "build the rolling retraining loop with no future leakage",
  "grade the gate without letting it see the defect flag", "check whether this gap could be luck".
- **Human verification:** every result was rerun on our own machine; we checked sanity points
  (W2 identical across conditions, AI beats physics, ~2% gate cost on clean data) and reviewed all
  code before committing.
