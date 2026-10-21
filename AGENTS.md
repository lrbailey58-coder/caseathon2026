# AGENTS.md

## Project Context
Submission for the **W&M AI Case-a-thon 2026** (*Unintentional Data Poisoning*). See `Architecture.md` for the system design and `Case-a-thon Prompt.pdf` for competition rules.

- **Domain**: Autonomous Coastal Vessel Traffic & Collision-Avoidance Advisory System using MarineCadastre AIS data.
- **Task**: Predict 15-minute-ahead vessel positions ($\text{Lat}_{t+15}, \text{Lon}_{t+15}$) evaluated by Displacement Error in meters (MAE and 95th percentile tail error).
- **Poisoning Mechanism**: Rolling retraining across 6 chronological windows ($W_1 \dots W_5$ train, $W_6$ unseen test) fed by a delayed, high-accuracy `TIER1_ANCHOR` stream and a fast, high-volume `TIER2_FAST` stream that degrades over time.
- **Countermeasure**: Training-time cross-source trust gating (`TIER1_ANCHOR` audits `TIER2_FAST`) + inference-time quantile uncertainty deferral to a Watch Officer Human-in-the-Loop (HITL) workflow.

## Repository Layout & Boundaries
- `backend/`: Target directory for all Python code (FastAPI server, DuckDB schema/queries, degradation injector, and offline ML pipeline).
  - *Note*: `pyproject.toml` was initially scaffolded with `uv_build` pointing at `src/caseathon2026/`. Place new Python modules in `backend/` and update `pyproject.toml` packaging/entrypoints accordingly when wiring the backend.
- `frontend/`: React 19 + TypeScript + Vite 8 + Tailwind CSS + React-Leaflet + Recharts dashboard.
- `Architecture.md`: Source of truth for system architecture, DuckDB table design, and UI layout.

## Commands

### Python (`uv`, Python 3.13)
Run from the repository root:
- Install / sync dependencies: `uv sync`
- Add a package: `uv add <package>`
- Run Python scripts or modules: `uv run python <script.py>`
- Run FastAPI dev server (once wired in `backend/`): `uv run uvicorn backend.main:app --reload`

### Frontend (`npm`, working directory: `frontend/`)
Run from `frontend/`:
- Start dev server: `npm run dev`
- Lint (uses `oxlint`, not ESLint): `npm run lint`
- Typecheck and production build: `npm run build` (runs `tsc -b && vite build`)
- Typecheck only: `npx tsc -b`

## Toolchain & Codebase Gotchas
- **Python version**: Pinned to `>=3.13` (`.python-version` is `3.13`), while `Architecture.md` mentions 3.12. Trust `pyproject.toml` (`3.13`).
- **TypeScript strictness**: `frontend/tsconfig.app.json` enforces `noUnusedLocals`, `noUnusedParameters`, `verbatimModuleSyntax` (use `import type` for type-only imports), and `erasableSyntaxOnly`. Path alias `@/*` resolves to `./src/*`.
- **Vite 8 API Proxy**: `frontend/vite.config.ts` proxies `/api/*` requests to `http://localhost:8000` in dev mode, and `frontend/src/App.tsx` includes a deterministic local simulation fallback (`src/lib/fallbackData.ts`) if the FastAPI server is offline.

## Domain & Experiment Invariants (Strict Competition Rules)
When implementing data pipelines, ML experiments, or UI views, preserve these rules from `Case-a-thon Prompt.pdf` and `Architecture.md`:
1. **8-Run Experimental Matrix**: Every experiment must evaluate all **4 conditions** (`NOMINAL`, `STALE`, `LOW_QUALITY`, `COMBINED`) $\times$ **2 mitigation states** (unmitigated vs. mitigated) across windows $W_1 \dots W_6$.
2. **Zero Future / Label Leakage**:
   - Preserve strict chronological ordering across windows $W_1 \dots W_6$; never randomly shuffle across time.
   - Keep complete vessel voyages together as the evaluation unit.
   - Injected defect flags in `injected_streams` must remain strictly hidden from the training and inference algorithms.
   - Held-back future vessel positions ($t+15$) must never leak into live or training features.
3. **Separate Latency Reporting**: Always track and display **model compute latency (ms)** separately from **human review/override duration (s)** in `experiment_metrics`, `hitl_audit_log`, and the UI.
4. **Preserve Valid Rare Events**: Do not engineer a guaranteed win—evaluate whether kinematic/trust gating falsely rejects valid sharp maneuvers (e.g., tugboat/pilot turns) and document this boundary case (Requirement 7).
5. **DuckDB Core Tables**: Maintain the four shared tables (`raw_reference_ais`, `injected_streams`, `experiment_metrics`, `hitl_audit_log`) and export machine-readable manifests (`manifest.json`, modification logs, and results).
