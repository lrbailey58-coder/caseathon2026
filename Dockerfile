# ============================================================================
# Stage 1: Build React + Vite Frontend
# ============================================================================
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./

ARG VITE_CARTO_API_KEY
ENV VITE_CARTO_API_KEY=${VITE_CARTO_API_KEY}

RUN npm run build

# ============================================================================
# Stage 2: FastAPI + DuckDB Unified Runtime
# ============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

WORKDIR /app

# Copy project configuration and install runtime dependencies into both `.venv` and system Python
COPY pyproject.toml uv.lock* README.md* ./
RUN uv venv --python 3.12 .venv \
    && uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" duckdb pydantic \
    && uv pip install --system fastapi "uvicorn[standard]" duckdb pydantic

ENV UV_NO_SYNC=1
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy backend source and DuckDB data directory
COPY backend/ ./backend/
COPY data/ ./data/

# Real ML results: the backend loads this file on start instead of its demo seed data
COPY results/dashboard_export_real.json ./results/dashboard_export_real.json

# Copy compiled React SPA build from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

EXPOSE 10000

CMD ["sh", "-c", "uv run uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
