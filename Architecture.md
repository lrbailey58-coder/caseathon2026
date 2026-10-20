### 1. System Definition & Architectural Blueprint

To satisfy **Requirement 1 (Define the Decision)** and **Requirement 5 (Explain the Mechanism)**, the system is framed around an **Autonomous Coastal Vessel Traffic & Collision-Avoidance Advisory System** operating in a high-traffic maritime corridor (e.g., Chesapeake Bay / Port of Virginia approaches).

* **The Decision & User:** A harbor Watch Officer or autonomous vessel routing controller uses 15-minute-ahead vessel position predictions ($\text{Lat}_{t+15}, \text{Lon}_{t+15}$) to detect lane incursions and issue course-correction advisories.


* **Evaluation Unit & Metric:** Complete, chronologically ordered vessel voyages segmented into 15-minute prediction horizons, evaluated via **Displacement Error in meters (Mean Absolute Error & 95th Percentile Tail Error)**.


* **The Training-Time Poisoning Mechanism:** The model retrains on a rolling weekly schedule using an ingestion buffer fed by two sources:


1. **Tier-1 Coastal Anchor Feed (Slow, High Accuracy):** Authoritative USCG terrestrial AIS receivers with high positional fidelity, arriving with a 30-to-60-minute batch delay and lower ping frequency.


2. **Tier-2 Commercial Aggregator Feed (Fast, High Volume, Degrading):** A real-time commercial stream that progressively floods the retraining buffer with **Stale Records** (cached prior vessel coordinates re-broadcast with fresh timestamps) and **Low-Quality Records** (corrupted Speed Over Ground `SOG` and Course Over Ground `COG` spikes).




* **The Countermeasure (Anchor-Guided Trust Gate + HITL Abstention):**
1. **Training-Time Cross-Source Validator:** Before each rolling retraining cycle, the pipeline retroactively aligns Tier-2 fast records against the delayed Tier-1 Anchor trajectory. It calculates a **Source Trust Weight** and filters out Tier-2 segments that violate kinematic constraints or diverge from the Tier-1 anchor.


2. **Inference-Time Uncertainty Deferral (Human-in-the-Loop):** Using a quantile regression ensemble, when prediction spread exceeds a safety radius or input staleness is high, the UI withholds autonomous execution and triggers a **Watch Officer Review Modal** with a 1-click kinematic fallback override.





---

### 2. Software Architecture

Because the team does not need to run heavy model retraining live during the 10-minute presentation, the architecture separates the **Offline Rolling-Experiment Engine** from the **Low-Latency FastAPI + React Web Application**, linked by a shared **DuckDB** analytical database.

```
+-----------------------------------------------------------------------------------+
|                        OFFLINE DATA & ML PIPELINE (Python / uv)                   |
|                                                                                   |
|  [MarineCadastre Bulk AIS] ---> [1. Ingestion & Voyage Grouper]                   |
|   (Untouched Parquet Ref)               |                                         |
|                                         v                                         |
|                             [2. Degradation Injector]                             |
|                     (Nominal | Stale | Low-Quality | Combined)                    |
|                                         |                                         |
|                        +----------------+----------------+                        |
|                        |                                 |                        |
|                        v (Unmitigated)                   v (Mitigated)            |
|           [Naive Rolling Retrainer]         [Anchor-Verified Trust Gate]          |
|                        |                    (Slow Feed Audits Fast Feed)          |
|                        |                                 |                        |
|                        |                    [Weighted Rolling Retrainer]          |
|                        +----------------+----------------+                        |
|                                         |                                         |
|                                         v                                         |
|                    [DuckDB Analytical Store & JSON Manifests]                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         LIVE DEMO APPLICATION STACK                               |
|                                                                                   |
|  [FastAPI Backend (uv)] <---> Queries Precomputed Rolling Runs & Live Inference   |
|            ^                                                                      |
|            | REST API (JSON)                                                      |
|            v                                                                      |
|  [React + Vite + Tailwind + shadcn/ui + Deck.gl Maritime Tactical Dashboard]      |
|   - Interactive Split-Map (Unmitigated Drift vs. Anchor-Verified Trajectory)      |
|   - Rolling Window Degradation Charts (W1 -> W6)                                  |
|   - Watch Officer HITL Override Workflow (Logs Human Review Time vs. Compute)     |
+-----------------------------------------------------------------------------------+

```

#### Recommended Tech Stack

* **Package & Environment Management:** `uv` (Python 3.12 workspace) + `npm` / `Vite` (Frontend).
* **Database Layer:** **DuckDB** (`duckdb` Python package). It queries MarineCadastre GeoParquet/CSV files directly in milliseconds without requiring a complex database server setup, enforces reproducible SQL schemas, and exports the required machine-readable results manifest.


* **ML Framework:** **LightGBM** or **XGBoost** (`scikit-learn` wrapper) Predicting $(\Delta \text{Lat}, \Delta \text{Lon})$ from a sliding window of vessel states, paired with Quantile Regression ($\alpha = 0.05, 0.50, 0.95$) to drive the Human-in-the-Loop uncertainty bounds.


* **Backend API:** **FastAPI** + **Pydantic** for clean REST endpoints.
* **Polished Frontend Stack:** **React (Vite + TypeScript)** + **Tailwind CSS** + **shadcn/ui** (gives an instant, sleek dark-mode command-center look) + **Deck.gl** or **React-Leaflet** (for glowing vessel tracks and uncertainty cones) + **Recharts** (for degradation timelines).



---

### 3. Team Roles (3-Person Split)

| Role | Title | Core Responsibilities | Primary Deliverables Owned |
| --- | --- | --- | --- |
| **Person 1** *(Database Lead)* | **Data & Database Architect** | • Download MarineCadastre AIS bulk files and build the DuckDB schema.

<br>

<br>• Group pings into leak-free chronological vessel voyages across 6 time windows ($W_1 \dots W_6$).

<br>

<br>• Build the **Degradation Injector** to generate the 4 required data conditions (Nominal, Stale-only, Low-Quality-only, Combined).

 | • Untouched Reference Copy & Data Manifest (`manifest.json`).

<br>

<br>• DuckDB database file (`maritime_case.duckdb`).<br>

<br>• Modification Log documenting exact defect ratios per window.

 |
| **Person 2** *(Undecided #1)* | **ML & Countermeasure Engineer** | • Build the feature engineering pipeline and rolling-window retraining loop ($W_1 \dots W_5$ train, $W_6$ unseen test).

<br>

<br>• Implement the **Cross-Source Anchor Gatekeeper** (using the slow Tier-1 feed to audit/weight the fast Tier-2 feed).

<br>

<br>• Run the $4 \times 2$ experimental matrix and log error, coverage, and rare-maneuver failure bounds.

 | • Rolling Retraining & Countermeasure Engine.<br>

<br>• Machine-readable results across all 8 conditions.

<br>

<br>• Diagnostic proof of mechanism & boundary failure case (e.g., sharp tugboat maneuvers falsely flagged as noise).

 |
| **Person 3** *(Undecided #2)* | **Full-Stack & Tactical UI Architect** | • Scaffold the FastAPI backend and React + `shadcn/ui` dark-mode dashboard.

<br>

<br>• Build the interactive maritime map showing vessel tracks, poisoned drift, and mitigated recovery.<br>

<br>• Build the **Watch Officer HITL Override widget** that measures human review latency separately from model compute time.

 | • Working Web Application + Setup README.

<br>

<br>• Recorded fallback video demonstration.

<br>

<br>• Interactive HITL walkthrough for the 10-minute presentation.

 |

---

### 4. Database & Experimental Matrix Design (For Person 1 & 2)

To satisfy the **Minimum Comparison Structure**, the DuckDB database will store four distinct tables so the React app can switch between any condition instantaneously during the live Q&A:

1. `raw_reference_ais`: Untouched subset of MarineCadastre data (e.g., 4–6 weeks around Norfolk/Chesapeake Bay, filtered to cargo/tanker/tug vessels in motion).


2. `injected_streams`: Contains `window_id` ($1 \dots 6$), `source_tier` (`TIER1_ANCHOR` vs `TIER2_FAST`), `condition` (`NOMINAL`, `STALE`, `LOW_QUALITY`, `COMBINED`), and injected defect flags (kept strictly hidden from the algorithm during training).


* **Stale Injection:** For increasing percentages of `TIER2_FAST` messages ($10\% \rightarrow 70\%$ across $W_2 \dots W_5$), overwrite current $(\text{Lat}, \text{Lon}, \text{SOG}, \text{COG})$ with values from $10\text{--}20$ minutes earlier in the same voyage while keeping the current timestamp.


* **Low-Quality Injection:** Inject Gaussian multipath jitter into $(\text{Lat}, \text{Lon})$ and random spikes into `SOG`/`COG` on `TIER2_FAST` records.




3. `experiment_metrics`: Stores the 8-run matrix ($4 \text{ conditions} \times 2 \text{ mitigation states}$) across every time window $W_1 \dots W_6$, recording Mean Displacement Error (m), 95th Percentile Error (m), Data Discarded (%), Decisions Deferred to Human (%), and Model Compute Latency (ms).


4. `hitl_audit_log`: Stores live or simulated Watch Officer interventions, capturing `alert_timestamp`, `decision_timestamp`, `human_review_seconds`, `action_taken` (`ACCEPT_ANCHOR_FALLBACK`, `MANUAL_WAYPOINT`, `DISMISS`), and operator rationale.



---

### 5. Day-by-Day 1-Week Execution Plan

| Day | Person 1 (Database & Data Pipeline) | Person 2 (ML & Countermeasure) | Person 3 (Full-Stack & React UI) |
| --- | --- | --- | --- |
| **Day 1**<br>

<br>*Setup & Scope Lock* | Initialize `uv` workspace. Download MarineCadastre bulk file (state whether CSV or GeoParquet and check license). Filter to 1 port region and verify clean voyage IDs.

 | Define feature vector ($\text{Lat}, \text{Lon}, \text{SOG}, \text{COG}, \Delta v, \Delta \theta$) and target ($\Delta \text{Lat}_{t+15}, \Delta \text{Lon}_{t+15}$). Write baseline LightGBM regressor on dummy data. | Initialize React + Vite + Tailwind + `shadcn/ui` repo and FastAPI skeleton. Build the static layout with a map canvas covering the selected port region. |
| **Day 2**<br>

<br>*Baseline & Schema* | Build DuckDB ingestion script. Partition data chronologically into Windows $W_1 \dots W_6$ ensuring complete voyages stay together. Generate the initial **Data Manifest**.

 | Train & evaluate the **Requirement 2 Reference Baseline** on $W_1$. Establish baseline MAE and 95th percentile error and verify zero future data leakage.

 | Connect FastAPI to DuckDB. Render real reference vessel trajectories from $W_1$ on the React map with playback controls. |
| **Day 3**<br>

<br>*Poisoning Injection* | Build the **Degradation Injector** for `STALE`, `LOW_QUALITY`, and `COMBINED` across $W_2 \dots W_5$. Split streams into `TIER1_ANCHOR` (20% volume, clean, delayed) and `TIER2_FAST` (80% volume, degrading).

 | Implement the **Unmitigated Rolling Retraining Loop** across $W_1 \dots W_5$ and test on $W_6$. Verify that training-time drift visibly degrades trajectory predictions over time.

 | Build the **Degradation Timeline Dashboard** (Recharts) showing data age, source volume mix, and rising prediction error across $W_1 \dots W_6$.

 |
| **Day 4**<br>

<br>*Countermeasure & HITL* | Write SQL views in DuckDB to pair delayed `TIER1_ANCHOR` points with `TIER2_FAST` histories for cross-source auditing, plus table schemas for `hitl_audit_log`. | Implement the **Anchor-Verified Trust Gate** (down-weighting/filtering corrupted Tier-2 data before retraining) + Quantile Uncertainty flags. Run all 8 conditions.

 | Build the **Watch Officer HITL Modal** in React: displays data age, source trust score, uncertainty cone on map, and a timed "Approve/Override" button that logs human review duration.

 |
| **Day 5**<br>

<br>*Limits & Tradeoffs* | Audit all tables for completeness. Export machine-readable results (`results.csv`/`json`) and finalize the modification log and data governance documentation.

 | Isolate **Requirement 7 (Failure/Boundary Case)**: find a real vessel executing a rapid valid turn (e.g., pilot boat/tug) that the kinematic gate falsely discards as "low quality".

 | Add a "Failure Mode & Tradeoff Inspector" tab in the UI to showcase the rare-event false positive and human override recovery during the presentation.

 |
| **Day 6**<br>

<br>*Polish & Fallback Video* | Run repeat-seed stability checks with Person 2 and document compute costs, NIST AI RMF governance roles, and pilot stop/go criteria.

 | Finalize diagnostic plots proving *why* the model drifted (e.g., feature importance shift / stale velocity bias pushing predictions off-course).

 | Polish UI transitions, add a 1-click "Demo Scenario Switcher" in the header, write the setup `README.md`, and **record the fallback demo video**.

 |
| **Day 7**<br>

<br>*Rehearsal* | Rehearse the 10-minute live presentation (3 mins problem/data architecture, 4 mins live UI + HITL walkthrough, 3 mins results matrix, limits & governance).

 | Verify all numbers match across slides, UI, and machine-readable manifests.

 | Test local offline startup from a clean terminal to guarantee zero network/API failures during the live talk.

 |

---

### 6. High-Impact UI Layout for the React App (Person 3's Target)

To make the React app look like an executive-grade product without taking weeks of custom CSS work, structure it as a single-screen **3-Pane Tactical View** using `shadcn/ui`:

1. **Top Control Bar (Scenario & Condition Switcher):**
* Dropdown 1: **Condition** (`Reference`, `Stale Only`, `Low-Quality Only`, `Combined`).


* Toggle: **Countermeasure** (`OFF: Naive Retraining` vs. `ON: Anchor-Verified + HITL`).


* Slider: **Rolling Time Window** (`Week 1: Nominal` $\rightarrow$ `Week 5: Severe Drift` $\rightarrow$ `Week 6: Unseen Test Period`).




2. **Center Canvas (Interactive Maritime Map - Deck.gl / Leaflet):**
* **Green Solid Line:** Ground-truth vessel track.


* **Red Dashed Line:** Unmitigated model prediction (visibly drifting out of the shipping channel as stale/noisy training data accumulates over Weeks 2–5).


* **Cyan Solid Line + Shaded Cone:** Mitigated model prediction + 95% uncertainty boundary.


3. **Right Sidebar (Telemetry, Metrics & Watch Officer HITL Panel):**
* **Top Card (Live Metrics):** Displays Primary Error (meters), Data Discarded (%), Model Latency (`14 ms`), and Avg Human Review Time (`8.4 s`) side-by-side.


* **Middle Card (Rolling Drift Chart):** Line chart showing error across $W_1 \dots W_6$ with vs. without the countermeasure.


* **Bottom Card (Watch Officer Action Queue - Requirement 4 Oversight):** Highlights a flagged vessel where feed staleness > threshold. Shows **Evidence** ("Tier-2 Fast Feed deviates 420m from Tier-1 Coastal Anchor; Staleness = 18 mins"), and gives the user two buttons: `[Defer to Coastal Anchor Kinematic Extrapolation]` or `[Manual Waypoint Override]`, with a live timer recording human verification effort.





Would you like me to generate the initial `uv` project structure, the DuckDB SQL schema for the 4 degradation conditions, or the React component wireframes so your team can start coding immediately?