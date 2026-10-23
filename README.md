# NEPTUNE-GUARD // Maritime AIS Drift & Oversight Console

**W&M AI Case-a-thon 2026 — Unintentional Data Poisoning (Stale & Low-Quality Training Streams)**  
**Authors:** L. R. Bailey (`lrbailey@wm.edu`), Jay Lavender, Francis Navarro-Lim

---

## 1. Executive Summary & Decision Context

**NEPTUNE-GUARD** is an Autonomous Coastal Vessel Traffic & Collision-Avoidance Advisory System operating in the **Port of Virginia / Chesapeake Bay Approach (Thimble Shoal Inbound Channel, 36.8°N–37.3°N, 76.6°W–75.8°W)**.

- **The Operational Decision & User:** A harbor **Watch Officer** (USCG Vessel Traffic Service / Port Operations) and automated channel-routing advisory controller rely on **15-minute-ahead vessel coordinate predictions** ($\text{Lat}_{t+15}, \text{Lon}_{t+15}$) to detect lateral channel excursions and issue timely course-correction advisories.
- **Evaluation Unit & Primary Metrics:** Complete, chronologically ordered vessel voyages segmented at 30-minute gaps (never randomly shuffled across time), evaluated by **Displacement Error in meters**:
  1. **Primary Metric:** Mean Absolute Displacement Error (**MAE**, meters) at $t+15\text{ min}$.
  2. **Tail Safety Metric:** **95th Percentile Displacement Error** (meters) to capture worst-case channel excursion risk.
- **The Unintentional Training-Time Poisoning Mechanism:**
  The forecasting model retrains across 6 chronological windows ($W_1 \dots W_5$ rolling training windows; $W_6$ held-out unseen evaluation period) using two maritime telemetry feeds:
  - **`TIER1_ANCHOR` (20% volume, clean, delayed):** Authoritative USCG terrestrial AIS receivers with high positional accuracy but a 30–60 minute batch ingestion delay.
  - **`TIER2_FAST` (80% volume, real-time, degrading):** High-frequency commercial aggregator stream that progressively degrades across $W_2 \dots W_6$ via:
    1. **Stale Records (`STALE`):** Re-broadcasting cached vessel positions/velocities from 10 pings (~12 minutes) earlier in the voyage under fresh timestamps ($15\% \rightarrow 70\%$ of fast-feed volume).
    2. **Low-Quality Records (`LOW_QUALITY`):** Multipath GPS coordinate jitter ($\sigma = 0.0005^\circ \rightarrow 0.0035^\circ$) plus Speed Over Ground (`SOG` $+5\text{–}15\text{ kn}$) and Course Over Ground (`COG` $+30^\circ\text{–}90^\circ$) spikes.
    3. **Combined Degradation (`COMBINED`):** Simultaneous staleness lag and multipath/kinematic sensor corruption.
- **The Countermeasure (Anchor-Guided Trust Gate + Quantile HITL Deferral):**
  1. **Training-Time Cross-Source Trust Gate:** Before rolling retraining on $W_k$, delayed `TIER1_ANCHOR` tracks retroactively audit `TIER2_FAST` observations using kinematic and positional divergence thresholds learned exclusively on clean baseline window $W_1$ (zero label or defect-flag leakage). Suspect fast-feed pings are dropped prior to LightGBM retraining (Tier-1 anchor pings are always kept).
  2. **Inference-Time Quantile Uncertainty Deferral (HITL):** Quantile LightGBM models ($\alpha \in \{0.05, 0.95\}$) estimate a 90% uncertainty range around each prediction. When its radius exceeds 1 nautical mile (1,852 m), autonomous execution is withheld and routed to the **Watch Officer HITL Oversight Queue**, logging human verification effort (seconds) strictly separately from model compute latency (milliseconds).

---

## 2. Repository Architecture

```text
caseathon2026/
├── backend/
│   ├── main.py                      # FastAPI server (/api/scenario, /api/matrix-summary, /api/hitl/*, /api/export/results)
│   ├── db.py                        # DuckDB schema; loads results/dashboard_export_real.json on start (demo seed only if missing)
│   ├── database/
│   │   ├── build_database.py        # NOAA MarineCadastre 2025 AIS ingestion & degradation injector
│   │   ├── manifest.json            # Dataset manifest (671,747 reference pings, 9,680 voyages, Sept 1 - Oct 12, 2025)
│   │   └── modification_log.json    # Exact per-window stale & noise injection counts
│   └── ml/                          # ML pipeline - full documentation in backend/ml/README.md
│       ├── check_data.py            # 22 data checks before training (PASS / WARN / FAIL)
│       ├── data.py                  # Leak-free DuckDB loader across W1..W6 (drops source glitches)
│       ├── features.py              # Kinematic feature engineering & t+15m target builder
│       ├── gate.py                  # Cross-source Tier-1 Anchor Trust Gate
│       ├── uncertainty.py           # Quantile regression uncertainty cones & deferral rules
│       ├── train_baseline.py        # W1 reference baseline training
│       ├── rolling.py               # 8-run (4 conditions x 2 mitigation states) rolling retraining loop
│       ├── grade_gate.py            # Grades the trust gate as a detector (catches, misses, false alarms)
│       ├── summary.py               # 8-run W6 table + bootstrap uncertainty check
│       ├── backfire.py              # Failure mode & boundary stress checks (sharp turns, short stale, GPS bias)
│       └── export_dashboard.py      # Exports results to results/dashboard_export_real.json (auto-loaded by the backend)
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # 3-pane tactical state controller + offline simulation fallback
│   │   ├── components/
│   │   │   ├── HeaderControls.tsx   # Top controls + 1-Click Live Presentation Storyboard + JSON Export
│   │   │   ├── MaritimeMap.tsx      # React-Leaflet tactical chart, uncertainty cones, OPERATOR VERIFIED animation
│   │   │   ├── TelemetrySidebar.tsx # Metrics grid, Recharts W1->W6 drift chart, HITL queue, Failure Mode card
│   │   │   └── ExperimentMatrixModal.tsx # Full 4x2 experimental comparison table across W1..W6
│   │   └── lib/fallbackData.ts      # Deterministic local fallback if FastAPI is offline
├── data/
│   └── maritime_case.duckdb         # Local DuckDB analytical store served by FastAPI
├── results/                         # Machine-readable JSON/Markdown outputs from the offline ML pipeline
├── Architecture.md                  # System design & architectural blueprint
└── pyproject.toml                   # Python 3.13 + uv dependency specification
```

---

## 3. Setup & Quickstart Instructions (Windows / macOS / Linux)

### Prerequisites
- **Python:** `>= 3.13` managed via [`uv`](https://docs.astral.sh/uv/)
- **Node.js:** `>= 20.x` (tested with Node 22/24) and `npm`

---

### Step 1: Backend Setup (`uv` + FastAPI + DuckDB)

Run all Python commands from the **repository root** (`caseathon2026/`):

#### Windows (PowerShell)
```powershell
# 1. Install Python 3.13 virtual environment and all dependencies
uv sync

# 2. Start the FastAPI + DuckDB tactical backend server on http://localhost:8000
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

#### macOS / Linux (Bash / Zsh)
```bash
# 1. Install Python 3.13 virtual environment and all dependencies
uv sync

# 2. Start the FastAPI + DuckDB tactical backend server on http://localhost:8000
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

> **Note:** On startup, `backend/main.py` automatically initializes `data/maritime_case.duckdb` via `init_and_seed_db()` if not already present and loads the real ML results from `results/dashboard_export_real.json` (built-in demo values are used only if that file is missing). `http://localhost:8000/api/health` shows which is live under `data_source.kind` (`real` or `demo`). Interactive Swagger API docs are available at `http://localhost:8000/docs`, and the machine-readable results bundle can be downloaded directly from `http://localhost:8000/api/export/results`.

---

### Step 2: Frontend Setup (`npm` + React 19 + Vite 8)

Open a second terminal and navigate to `frontend/`:

#### Windows / macOS / Linux
```bash
cd frontend

# 1. Install Node packages
npm install

# 2. (Optional) Configure Carto basemap key in frontend/.env.local
# Works out-of-the-box without a key using Carto Dark Matter public tiles
# VITE_CARTO_API_KEY=your_optional_key

# 3. Start the Vite development server on http://localhost:5173
npm run dev
```

To run strict TypeScript verification, `oxlint`, and a production build:
```bash
cd frontend
npm run lint
npm run build
```

> **Zero-Downtime Offline Fallback:** `frontend/vite.config.ts` proxies `/api/*` requests to `http://localhost:8000`. If the FastAPI backend is offline, the React application automatically switches to `LOCAL SIM FALLBACK` mode (`src/lib/fallbackData.ts`) so the 1-Click Presentation Storyboard, HITL override animations, and JSON export continue to work without interruption.

---

### Step 3: Re-Running the Offline ML & Data Pipeline (Optional)

Precomputed results are committed in `results/`, so the site works without this step. Re-running needs Person 1's `backend/maritime_case.duckdb` (not in git; rebuild with `uv run python backend/database/build_database.py`). Run from the repository root, in this order (about 15 minutes); details in [`backend/ml/README.md`](backend/ml/README.md):

```bash
uv run python -m backend.ml.check_data                    # data checks: must show 0 FAIL
uv run python -m backend.ml.train_baseline --source real  # W1 reference baseline vs physics
uv run python -m backend.ml.rolling                       # 4 conditions x gate off/on, W1..W5 -> W6
uv run python -m backend.ml.grade_gate                    # trust gate catches / misses / false alarms
uv run python -m backend.ml.summary                       # 8-run W6 table + bootstrap uncertainty
uv run python -m backend.ml.backfire                      # failure cases: sharp turns, new corruptions
uv run python -m backend.ml.export_dashboard              # results/dashboard_export_real.json for the site
```

Commit `results/` afterwards so the hosted site shows the new numbers.

---

## 4. Package Dependencies & Versions

### Python Backend & ML Stack (`pyproject.toml`, Python `>=3.13`)
| Package | Version Constraint | Role in System |
| :--- | :--- | :--- |
| `python` | `>=3.13` | Core runtime environment (managed via `uv`) |
| `fastapi` | `>=0.156.4` | REST API server for tactical scenario queries, HITL persistence, and JSON manifest export |
| `uvicorn[standard]` | `>=0.54.0` | ASGI server hosting FastAPI |
| `pydantic` | `>=2.14.1` | Strict request/response schema validation |
| `duckdb` | `>=2.0.0` | Embedded analytical SQL database (`maritime_case.duckdb`) |
| `lightgbm` | `>=4.7.0` | Gradient boosted decision trees for $(\Delta\text{Lat}, \Delta\text{Lon})$ and quantile uncertainty cones |
| `scikit-learn` | `>=1.9.1` | Evaluation metrics and pipeline utilities |
| `pandas` | `>=3.1.0` | Chronological voyage windowing and feature alignment |
| `numpy` | `>=2.5.4` | Vectorized kinematic math and deterministic seeded injection |
| `requests` | `>=2.34.2` | NOAA MarineCadastre 2025 AIS bulk stream ingestion |
| `zstandard` | `>=0.25.0` | High-speed `.csv.zst` decompression for NOAA AIS archives |

### Frontend Tactical UI Stack (`frontend/package.json`)
| Package | Version Constraint | Role in System |
| :--- | :--- | :--- |
| `react` / `react-dom` | `^19.3.1` | Component UI rendering |
| `typescript` | `~6.0.3` | Strict static type checking (`verbatimModuleSyntax`, `noUnusedLocals`) |
| `vite` | `^8.3.4` | Frontend bundler and `/api` dev proxy |
| `tailwindcss` | `^3.4.19` | Dark-mode tactical command-center styling |
| `leaflet` / `react-leaflet` | `^1.9.4` / `^5.0.0` | Interactive Norfolk shipping channel map, trajectory polylines, and uncertainty cones |
| `recharts` | `^3.10.1` | Rolling retraining drift timeline charts ($W_1 \rightarrow W_6$) |
| `lucide-react` | `^1.60.0` | Maritime and telemetry iconography |
| `oxlint` | `^1.88.0` | High-speed linter enforcing React hooks and TypeScript hygiene |

---

## 5. Live Presentation Storyboard (4-Minute Demo Walkthrough)

The top header includes a **Live Presentation Storyboard** bar with 4 preset buttons designed for the 10-minute judge walkthrough:

1. **`[1. Baseline Reference (W1)]`** (`NOMINAL`, `W1`, Countermeasure `OFF`):  
   Demonstrates baseline trajectory accuracy on clean MarineCadastre AIS data prior to feed degradation.
2. **`[2. Unmitigated Drift (W5 Combined)]`** (`COMBINED`, `W5`, Countermeasure `OFF`):  
   Shows how weekly retraining on stale + multipath-corrupted `TIER2_FAST` data makes the unmitigated $t+15\text{m}$ predictions worse (W6: 707 m vs 667 m on clean data; cargo 608 m vs 543 m), even though the model keeps getting more data.
3. **`[3. Countermeasure Recovery (W6 Unseen)]`** (`COMBINED`, `W6`, Countermeasure `ON`):  
   Demonstrates recovery on the held-out unseen test window $W_6$ using the Tier-1 Anchor Trust Gate (707 m -> 665 m; cargo 608 m -> 546 m) and displays the 90% uncertainty range.
4. **`[4. Boundary Failure Case (Tugboat Maneuver)]`** (`HARBOR_TUG_RARE_MANEUVER`):  
   Loads a real tug voyage from the AIS data (picked by a fixed rule: the tug job whose valid sharp-turn reports the gate most often wrongly flags on clean data). Its reports curve away from the straight line the Trust Gate draws between Tier-1 anchor reports, so the gate discards valid data; the map shows the 30 minutes around its sharpest turn. The Watch Officer can confirm the turn with **`[Manual Waypoint Override]`**, which badges the vessel **`OPERATOR VERIFIED`** and persists the intervention to DuckDB.

---

## 6. Responsible AI, Human Oversight & Governance

In alignment with the **NIST AI Risk Management Framework (AI RMF 1.0)** and maritime safety standards, NEPTUNE-GUARD enforces strict operational boundaries between automated data hygiene and human command authority:

### 6.1 Human-in-the-Loop (HITL) User & Review Triggers
- **Designated Oversight User:** **USCG Vessel Traffic Service (VTS) Watch Officer** (Norfolk / Hampton Roads Sector).
- **Review Trigger (Automated Abstention):** a prediction is routed to the Watch Officer when its 90% uncertainty range has a radius above **1 nautical mile (1,852 m)**. In W6 this is about 19% of predictions (cargo about 9%, tugs about 24%), and those predictions really are the worse ones: about 1,500 m average error versus about 470 m for the ones the AI keeps.
- **Where the Trust Gate acts:** the gate filters the fast feed at **training time** (before each weekly retraining), using the delayed Tier-1 anchor feed. It does not block live predictions.
- **Strict Latency Accounting:** Model inference compute time (about 6 ms per vessel, about 14 ms including the uncertainty models, measured on our laptop at export time) is tracked and displayed strictly separately from Watch Officer verification and override duration (measured live with the review timer) in `hitl_audit_log`, `experiment_metrics`, and the exported JSON manifest.

### 6.2 Pilot Data-Quality Owner & Governance Roles
- **Pilot Data-Quality & Model Risk Owner:** **VTS Maritime Telemetry Systems Lead (Port of Virginia Joint Operations)**, responsible for weekly audits of `TIER2_FAST` rejection rates, `TIER1_ANCHOR` batch latency health, and retraining buffer drift before any rolling model weight update is promoted.
- **Pre-Pilot Governance Checkpoint for Rare Maneuvers (Failure Mode Mitigation):**
  Because kinematic trust gating creates an adverse tradeoff on legitimate sharp turns (clean-data false-alarm rate of **21.0%** on tugboats turning $>60^\circ/\text{min}$, **46.1%** on cargo vessels turning $20^\circ\text{–}60^\circ/\text{min}$ and **88.5%** above $60^\circ/\text{min}$, versus about 1% when steady), pilot deployment requires:
  1. **Turn-Aware Gate (next step, owner: ML lead):** compare fast-feed reports against a curved path (anchor positions plus their headings) instead of a straight line. Checkpoint: sharp-turn false alarms on clean data below 10% with no loss of poisoning recovery at W6.
  2. **Secondary Out-of-Band Verification:** Watch Officer VHF Channel 13 / radar confirmation for predictions routed to review before any automated advisory is issued.

### 6.3 Pilot Stop / Go Criteria
Before transitioning NEPTUNE-GUARD from shadow-mode evaluation to active Watch Officer advisory operations in the Norfolk corridor, the system must satisfy four quantitative gates across 4 consecutive rolling weeks:

| Governance Dimension | GO Criterion (Shadow -> Active Pilot) | STOP / Rollback Trigger (Immediate Freeze to W1 Baseline) |
| :--- | :--- | :--- |
| **1. Unseen Horizon Accuracy (Cargo Cohort)** | Mitigated $t+15\text{m}$ MAE remains within $\le 5\%$ of the clean-data model on the same week (observed W6: 665 m vs 667 m all ships; 546 m vs 543 m cargo). | Mitigated MAE degrades by $> 10\%$ vs. $W_1$ baseline or 95th percentile tail error exceeds channel safety bounds. |
| **2. Clean-Data False-Alarm Cost** | Trust Gate false-positive discard rate on verified clean `TIER1_ANCHOR` overlap remains $< 3.0\%$ overall (observed: `1.85%`). | False-positive rejection on clean traffic exceeds $5.0\%$ or degrades clean-data MAE by $> 25\text{ m}$. |
| **3. Watch Officer Cognitive Workload** | Deferred decisions routed to HITL remain $\le 20\%$ of active transits (`18.9%` observed in $W_6$ Combined) with mean human verification time $< 15\text{ s}$. | Deferral rate exceeds $30\%$ of transits for $> 2\text{ hours}$ (alert fatigue risk) or mean human review time exceeds $30\text{ s}$. |
| **4. Unseen Bias & Anchor Availability** | `TIER1_ANCHOR` feed availability $\ge 98\%$ with batch delay $\le 60\text{ min}$; static GPS bias audit drift $< 100\text{ m}$. | `TIER1_ANCHOR` outage $> 90\text{ min}$ or detection of uncalibrated static receiver bias ($> 250\text{ m}$ systematic offset, where the gate caught only `7.9%` of a simulated 300 m offset). |

---

## 7. AI Tool & Prompt Disclosure

In compliance with the W&M AI Case-a-thon 2026 rules regarding transparency and generative AI usage:

- **AI Tools Used:**
  - **OpenCode / Anthropic Claude & Google Gemini Coding Agents** (used for full-stack scaffolding, FastAPI + DuckDB integration, React-Leaflet visualization components, and documentation formatting).
  - **Anthropic Claude (Cowork, Claude desktop app)** for the ML pipeline (`backend/ml/`): writing and testing code, finding data problems and explaining results. Details in [`backend/ml/README.md`](backend/ml/README.md), section 10.
  - **AI inside the system:** only the LightGBM models trained by our code; no generative AI makes predictions or decisions at run time.
- **Representative Prompts & Human Engineering Workflow:**
  1. *Architecture & Pipeline Design:* Prompts were used to structure the separation between the offline MarineCadastre AIS ingestion/ML pipeline (`backend/database/build_database.py`, `backend/ml/*`) and the low-latency FastAPI + DuckDB tactical server (`backend/main.py`, `backend/db.py`).
  2. *Leak-Free Experimental Protocol:* Team members specified and verified all competition invariants—specifically enforcing voyage-level chronological partitioning (`NTILE(6)` by voyage start time), hiding `is_injected_defect` flags from training/inference, learning Trust Gate limits strictly on clean window $W_1$, and evaluating on unseen window $W_6$.
  3. *Failure Mode & HITL Instrumentation:* Prompts were used to wire the `hitl_audit_log` persistence endpoints (`/api/hitl/override`, `/api/hitl/logs`), the `HARBOR_TUG_RARE_MANEUVER` boundary preset (a real tug example chosen by a fixed rule), the 1-Click Live Presentation Storyboard bar, and `/api/export/results`.
- **Verification & Validation:** All statistical claims, 8-run experimental matrix numbers, bootstrap confidence intervals (`results/experiment_summary_real.json`), and failure-mode stress tests (`results/backfire_checks_real.json`) were executed and verified against real NOAA MarineCadastre 2025 AIS data (`671,747` pings across `9,680` Chesapeake Bay voyages, Sept 1 - Oct 12, 2025).
