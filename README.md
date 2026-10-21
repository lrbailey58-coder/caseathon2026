# NEPTUNE-GUARD // Autonomous Maritime AIS Drift & Oversight System

Submission for the **W&M AI Case-a-thon 2026** (*Unintentional Data Poisoning*).

**NEPTUNE-GUARD** is a tactical coastal vessel traffic and collision-avoidance advisory system for the Chesapeake Bay / Port of Virginia (Thimble Shoal Channel) approach. It predicts 15-minute-ahead vessel trajectories ($\text{Lat}_{t+15}, \text{Lon}_{t+15}$) across rolling weekly retraining windows ($W_1 \dots W_6$) and defends against unintentional training-time data poisoning when a high-volume commercial AIS stream (`TIER2_FAST`) degrades with stale records and multipath sensor noise.

See [`Architecture.md`](./Architecture.md) for the full system blueprint, experimental matrix, and team responsibilities.

---

## Prerequisites

1. **Python `>=3.13` & [`uv`](https://docs.astral.sh/uv/)** (for the FastAPI + DuckDB backend and offline ML pipeline)
   - Verify with `uv --version` and `uv run python --version`.
2. **Node.js `>=20` & `npm`** (for the React 19 + TypeScript + Vite 8 frontend)
   - Verify with `node --version` and `npm --version`.
3. **CARTO Basemaps API Key (Free)**
   - CARTO requires an API key (`?key=YOUR_KEY`) to serve raster (`dark_all`) map tiles without an *"API KEY REQUIRED"* watermark.
   - Request a free non-commercial key (no account required, delivered via email) at **[carto.com/basemaps/apikey](https://carto.com/basemaps/apikey/)**.

---

## Quickstart: Running the Website Locally

The application runs as two processes during local development: the **FastAPI backend** (`http://localhost:8000`) and the **Vite React frontend** (`http://localhost:5173`, which proxies `/api/*` requests to port `8000`).

> **Offline Fallback Note:** If the FastAPI server is not running, the frontend automatically falls back to a deterministic local simulation (`frontend/src/lib/fallbackData.ts`) and displays a `LOCAL SIM FALLBACK` badge in the top header.

### 1. Configure the Frontend Map API Key

Copy `frontend/.env.example` to `frontend/.env.local` and paste your CARTO API key:

```powershell
cd frontend
Copy-Item .env.example .env.local
```

Edit `frontend/.env.local`:

```dotenv
# Paste your CARTO Raster (PNG) API key below
VITE_CARTO_API_KEY=your_carto_api_key_here
```

*(Note: `.env.local` is gitignored so your personal key is never committed. In `frontend/src/components/MaritimeMap.tsx`, this key is appended to the CARTO tile URL as `?key=${VITE_CARTO_API_KEY}`.)*

### 2. Start the FastAPI Backend (Terminal 1)

From the **repository root** (`caseathon2026/`):

```powershell
# Install/sync Python dependencies into .venv
uv sync

# Start the FastAPI server on http://localhost:8000
uv run uvicorn backend.main:app --reload --port 8000
```

- Health check endpoint: `http://localhost:8000/api/health`
- Interactive Swagger API docs: `http://localhost:8000/docs`

### 3. Start the React Frontend (Terminal 2)

From the **`frontend/`** directory:

```powershell
cd frontend

# Install Node dependencies
npm install

# Start the Vite dev server on http://localhost:5173
npm run dev
```

Open **`http://localhost:5173`** in your browser. If you just added or changed `VITE_CARTO_API_KEY` in `frontend/.env.local`, restart `npm run dev` and perform a hard refresh (`Ctrl+F5` or `Ctrl+Shift+R`) so the browser clears any cached placeholder map tiles.

### Frontend Quality Checks

Run from `frontend/`:

```powershell
npm run lint      # Lint with oxlint
npx tsc -b        # Strict TypeScript typecheck
npm run build     # Typecheck + production bundle into frontend/dist/
```

---

## Updating the Backend to Use Real MarineCadastre AIS Data

Currently, `backend/main.py` serves deterministic scenario trajectories and error progression tables (`ERROR_PROGRESSION` and `_build_tracks()`) alongside an in-memory `HITL_AUDIT_LOG` so the UI can be tested end-to-end immediately.

To transition the backend from deterministic simulation data to real **MarineCadastre AIS** data and precomputed rolling-retraining outputs stored in **DuckDB** (`maritime_case.duckdb`), follow these steps:

### 1. Build the DuckDB Analytical Store (`maritime_case.duckdb`)

Create an offline ingestion and experiment pipeline (e.g., `backend/ingest.py`, `backend/injector.py`, and `backend/train_rolling.py`) that populates the four core tables defined in [`Architecture.md`](./Architecture.md):

1. **`raw_reference_ais`**:
   - Ingest untouched MarineCadastre AIS bulk data (GeoParquet or CSV) filtered to the Chesapeake Bay / Norfolk corridor.
   - Group pings chronologically into complete vessel voyages (`voyage_id`) and partition into 6 leak-free chronological windows (`window_id` $1 \dots 6$) without splitting voyages across windows.
   - Export `manifest.json` documenting the untouched reference dataset.
2. **`injected_streams`**:
   - Split voyages into `TIER1_ANCHOR` (delayed 30–60 min, high-accuracy USCG terrestrial reference) and `TIER2_FAST` (real-time commercial stream).
   - Run the degradation injector across windows $W_2 \dots W_5$ to generate all 4 required conditions: `NOMINAL`, `STALE` (cached historical coordinates re-broadcast with fresh timestamps), `LOW_QUALITY` (Gaussian GPS multipath jitter + `SOG`/`COG` spikes), and `COMBINED`.
   - Keep injected defect flags strictly hidden from the training and inference pipelines.
3. **`experiment_metrics`**:
   - Train the rolling regression model ($W_1 \dots W_5$ rolling train, $W_6$ unseen evaluation) across all **8 runs** ($4 \text{ conditions} \times 2 \text{ mitigation states}$: unmitigated vs. Anchor-Verified Trust Gate + quantile uncertainty deferral).
   - Store per-window metrics: `condition`, `mitigated`, `window`, `displacement_mae_m`, `tail_95_error_m`, `data_discarded_pct`, `decisions_deferred_pct`, and `model_compute_latency_ms`.
   - Store representative voyage track coordinates (`ground_truth_track`, `unmitigated_prediction_track`, `mitigated_prediction_track`, and 95% quantile `uncertainty_polygon`) and vessel alert metadata (`mmsi`, `vessel_name`, `tier1_vs_tier2_divergence_m`, `staleness_minutes`, `trust_score`, `explanation`) in a companion DuckDB table or JSON column.
4. **`hitl_audit_log`**:
   - Persist Watch Officer override actions (`audit_id`, `mmsi`, `action`, `review_duration_seconds`, `condition`, `window`, `logged_at_utc`, `model_compute_latency_ms`) so human review latency (seconds) is permanently tracked separately from model compute latency (milliseconds).

### 2. Wire `backend/main.py` Endpoints to DuckDB

The FastAPI Pydantic response schemas in `backend/main.py` (`ScenarioResponse`, `WindowMetrics`, `TimelinePoint`, `HitlAlertPayload`, and `HitlOverrideResponse`) already match the exact JSON contract expected by the React frontend (`frontend/src/types/maritime.ts`). No frontend code changes are required when swapping in real data:

1. **Database Connection**:
   - Point `duckdb.connect("maritime_case.duckdb")` (or configure the path via an environment variable) in `backend/main.py` instead of `duckdb.connect(":memory:")`.
2. **`GET /api/health`**:
   - Verify that `maritime_case.duckdb` exists and that all four required tables (`raw_reference_ais`, `injected_streams`, `experiment_metrics`, `hitl_audit_log`) are present and populated.
3. **`GET /api/scenario`**:
   - Replace the static `ERROR_PROGRESSION` dictionary lookup and `_build_tracks()` helper with parameterized DuckDB SQL queries filtering by `condition`, `mitigated`, and `window` ($1 \dots 6$).
   - Query the 6-window `timeline_series` (`W1`–`W6`) directly from `experiment_metrics` and `injected_streams`.
   - Compute `avg_human_review_s` via `SELECT AVG(review_duration_seconds) FROM hitl_audit_log`.
4. **`POST /api/hitl/override`**:
   - Replace `HITL_AUDIT_LOG.append(entry)` with an `INSERT INTO hitl_audit_log VALUES (...)` query so Watch Officer interventions persist across server restarts.
