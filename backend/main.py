from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.db import (
    CONDITION_LABELS,
    DB_PATH,
    WINDOW_LABELS,
    ConditionType,
    build_explanation,
    get_connection,
    init_and_seed_db,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_and_seed_db()
    yield


app = FastAPI(
    title="NEPTUNE-GUARD // Maritime AIS Drift & Oversight API",
    version="0.2.0",
    description="Day 2 Tactical API for W&M Case-a-thon 2026: Unintentional Data Poisoning (DuckDB-backed)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WindowMetrics(BaseModel):
    displacement_mae_m: float
    tail_95_error_m: float
    data_discarded_pct: float
    decisions_deferred_pct: float
    model_compute_latency_ms: float
    avg_human_review_s: float


class TimelinePoint(BaseModel):
    window: str
    unmitigated_mae: float
    mitigated_mae: float
    tier2_corrupted_volume_pct: float
    avg_data_staleness_min: float


class HitlAlertPayload(BaseModel):
    vessel_name: str
    mmsi: str
    tier1_vs_tier2_divergence_m: float
    staleness_minutes: float
    trust_score: float
    requires_human_review: bool
    explanation: str


class ScenarioResponse(BaseModel):
    condition: ConditionType
    mitigated: bool
    window: int
    window_label: str
    ground_truth_track: list[tuple[float, float]]
    unmitigated_prediction_track: list[tuple[float, float]]
    mitigated_prediction_track: list[tuple[float, float]]
    uncertainty_polygon: list[tuple[float, float]]
    metrics: WindowMetrics
    timeline_series: list[TimelinePoint]
    hitl_alert: HitlAlertPayload


class MatrixRunEntry(BaseModel):
    run_id: int
    condition: ConditionType
    condition_label: str
    mitigated: bool
    defense_label: str
    window: int
    window_label: str
    displacement_mae_m: float
    tail_95_error_m: float
    data_discarded_pct: float
    decisions_deferred_pct: float
    recovery_gain_pct: float
    model_compute_latency_ms: float


class MatrixSummaryResponse(BaseModel):
    window: int
    window_label: str
    avg_human_review_s: float
    runs: list[MatrixRunEntry]


class HitlOverrideRequest(BaseModel):
    mmsi: str = Field(..., examples=["367591240"])
    action: Literal["DEFER_TO_ANCHOR_KINEMATIC", "MANUAL_COURSE_OVERRIDE"]
    review_duration_seconds: float = Field(..., ge=0.0)
    condition: ConditionType
    window: int = Field(..., ge=1, le=6)


class HitlOverrideAuditEntry(HitlOverrideRequest):
    audit_id: int
    logged_at_utc: str
    model_compute_latency_ms: float


class HitlOverrideResponse(BaseModel):
    status: str
    message: str
    audit_entry: HitlOverrideAuditEntry
    total_interventions_logged: int
    avg_human_review_s: float


def _parse_latlon_json(raw_json: str) -> list[tuple[float, float]]:
    parsed = json.loads(raw_json)
    return [(float(pt[0]), float(pt[1])) for pt in parsed]


@app.get("/api/health")
def get_health() -> dict[str, object]:
    init_and_seed_db()
    duckdb_ready = False
    duckdb_version = "unknown"
    logged_interventions = 0
    metrics_rows = 0

    try:
        conn = get_connection()
        try:
            ver_row = conn.execute("SELECT version()").fetchone()
            if ver_row:
                duckdb_version = str(ver_row[0])
            audit_row = conn.execute("SELECT COUNT(*) FROM hitl_audit_log").fetchone()
            if audit_row:
                logged_interventions = int(audit_row[0])
            m_row = conn.execute("SELECT COUNT(*) FROM experiment_metrics").fetchone()
            if m_row:
                metrics_rows = int(m_row[0])
            duckdb_ready = True
        finally:
            conn.close()
    except Exception:
        duckdb_ready = False

    return {
        "status": "ONLINE",
        "service": "neptune-guard-api",
        "duckdb_ready": duckdb_ready,
        "duckdb_version": duckdb_version,
        "database_path": str(DB_PATH),
        "experiment_metrics_rows": metrics_rows,
        "region": "Port of Virginia / Chesapeake Bay Approach",
        "logged_hitl_interventions": logged_interventions,
    }


@app.get("/api/scenario", response_model=ScenarioResponse)
def get_scenario(
    condition: ConditionType = Query("COMBINED", description="Data degradation condition"),
    mitigated: bool = Query(True, description="Whether Anchor-Verified Trust Gate + HITL is active"),
    window: int = Query(5, ge=1, le=6, description="Rolling retraining window (1=W1 .. 6=W6)"),
) -> ScenarioResponse:
    init_and_seed_db()
    conn = get_connection()
    try:
        # 1. Query selected scenario metrics from experiment_metrics
        metrics_row = conn.execute(
            """
            SELECT
                displacement_mae_m,
                tail_95_error_m,
                data_discarded_pct,
                decisions_deferred_pct,
                tier2_corrupted_volume_pct,
                avg_data_staleness_min,
                model_compute_latency_ms
            FROM experiment_metrics
            WHERE condition = ? AND mitigated = ? AND window_id = ?
            LIMIT 1
            """,
            (condition, mitigated, window),
        ).fetchone()

        if not metrics_row:
            raise HTTPException(
                status_code=404,
                detail=f"No experiment_metrics found for condition={condition}, mitigated={mitigated}, window={window}",
            )

        (
            mae,
            tail_95,
            discarded_pct,
            deferred_pct,
            corrupted_pct,
            staleness_min,
            model_latency_ms,
        ) = (
            float(metrics_row[0]),
            float(metrics_row[1]),
            float(metrics_row[2]),
            float(metrics_row[3]),
            float(metrics_row[4]),
            float(metrics_row[5]),
            float(metrics_row[6]),
        )

        # 2. Query vessel trajectory and alert telemetry from vessel_trajectories
        traj_row = conn.execute(
            """
            SELECT
                mmsi,
                vessel_name,
                ground_truth_json,
                unmitigated_pred_json,
                mitigated_pred_json,
                uncertainty_poly_json,
                tier1_divergence_m,
                staleness_min,
                trust_score
            FROM vessel_trajectories
            WHERE condition = ? AND mitigated = ? AND window_id = ?
            LIMIT 1
            """,
            (condition, mitigated, window),
        ).fetchone()

        if not traj_row:
            raise HTTPException(
                status_code=404,
                detail=f"No vessel_trajectories found for condition={condition}, mitigated={mitigated}, window={window}",
            )

        mmsi = str(traj_row[0])
        vessel_name = str(traj_row[1])
        gt_track = _parse_latlon_json(str(traj_row[2]))
        unmit_track = _parse_latlon_json(str(traj_row[3]))
        mit_track = _parse_latlon_json(str(traj_row[4]))
        cone_poly = _parse_latlon_json(str(traj_row[5]))
        divergence_m = float(traj_row[6])
        traj_staleness_min = float(traj_row[7])
        trust_score = float(traj_row[8])

        # 3. Query aggregate average of review_duration_seconds from hitl_audit_log
        avg_review_row = conn.execute(
            "SELECT COALESCE(AVG(review_duration_seconds), 0.0) FROM hitl_audit_log"
        ).fetchone()
        avg_review_s = round(float(avg_review_row[0]), 1) if avg_review_row else 0.0

        # 4. Query rolling timeline series W1..W6 for this condition (joining unmitigated & mitigated MAE)
        timeline_rows = conn.execute(
            """
            SELECT
                u.window_id,
                u.displacement_mae_m AS unmitigated_mae,
                m.displacement_mae_m AS mitigated_mae,
                u.tier2_corrupted_volume_pct,
                u.avg_data_staleness_min
            FROM experiment_metrics u
            JOIN experiment_metrics m
              ON u.condition = m.condition
             AND u.window_id = m.window_id
            WHERE u.condition = ?
              AND u.mitigated = FALSE
              AND m.mitigated = TRUE
            ORDER BY u.window_id ASC
            """,
            (condition,),
        ).fetchall()

        timeline_series = [
            TimelinePoint(
                window=f"W{int(r[0])}",
                unmitigated_mae=float(r[1]),
                mitigated_mae=float(r[2]),
                tier2_corrupted_volume_pct=float(r[3]),
                avg_data_staleness_min=float(r[4]),
            )
            for r in timeline_rows
        ]
    finally:
        conn.close()

    requires_review = condition != "NOMINAL" and window >= 2
    explanation = build_explanation(
        condition=condition,
        window_id=window,
        staleness_min=staleness_min,
        corrupted_pct=corrupted_pct,
        divergence_m=divergence_m,
    )

    return ScenarioResponse(
        condition=condition,
        mitigated=mitigated,
        window=window,
        window_label=WINDOW_LABELS.get(window, f"W{window}"),
        ground_truth_track=gt_track,
        unmitigated_prediction_track=unmit_track,
        mitigated_prediction_track=mit_track,
        uncertainty_polygon=cone_poly,
        metrics=WindowMetrics(
            displacement_mae_m=mae,
            tail_95_error_m=tail_95,
            data_discarded_pct=discarded_pct,
            decisions_deferred_pct=deferred_pct,
            model_compute_latency_ms=model_latency_ms,
            avg_human_review_s=avg_review_s,
        ),
        timeline_series=timeline_series,
        hitl_alert=HitlAlertPayload(
            vessel_name=vessel_name,
            mmsi=mmsi,
            tier1_vs_tier2_divergence_m=divergence_m,
            staleness_minutes=traj_staleness_min,
            trust_score=trust_score,
            requires_human_review=requires_review,
            explanation=explanation,
        ),
    )


@app.get("/api/matrix-summary", response_model=MatrixSummaryResponse)
def get_matrix_summary(
    window: int = Query(6, ge=1, le=6, description="Evaluation window to compare across all 8 runs (default 6=W6 Unseen Test Period)"),
) -> MatrixSummaryResponse:
    init_and_seed_db()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                m.condition,
                m.mitigated,
                m.window_id,
                m.displacement_mae_m,
                m.tail_95_error_m,
                m.data_discarded_pct,
                m.decisions_deferred_pct,
                m.model_compute_latency_ms,
                u.displacement_mae_m AS unmitigated_mae_m
            FROM experiment_metrics m
            JOIN experiment_metrics u
              ON m.condition = u.condition
             AND m.window_id = u.window_id
             AND u.mitigated = FALSE
            WHERE m.window_id = ?
            ORDER BY
                CASE m.condition
                    WHEN 'NOMINAL' THEN 1
                    WHEN 'STALE' THEN 2
                    WHEN 'LOW_QUALITY' THEN 3
                    WHEN 'COMBINED' THEN 4
                    ELSE 5
                END,
                m.mitigated ASC
            """,
            (window,),
        ).fetchall()

        avg_review_row = conn.execute(
            "SELECT COALESCE(AVG(review_duration_seconds), 0.0) FROM hitl_audit_log"
        ).fetchone()
        avg_review_s = round(float(avg_review_row[0]), 1) if avg_review_row else 0.0
    finally:
        conn.close()

    runs: list[MatrixRunEntry] = []
    for idx, row in enumerate(rows, start=1):
        cond = str(row[0])  # type: ignore[assignment]
        cond_typed: ConditionType = cond  # type: ignore[assignment]
        is_mitigated = bool(row[1])
        win_id = int(row[2])
        mae = float(row[3])
        tail_95 = float(row[4])
        discarded = float(row[5])
        deferred = float(row[6])
        latency_ms = float(row[7])
        unmit_mae = float(row[8])

        if is_mitigated and unmit_mae > 0:
            recovery_gain_pct = round(((unmit_mae - mae) / unmit_mae) * 100.0, 1)
        else:
            recovery_gain_pct = 0.0

        runs.append(
            MatrixRunEntry(
                run_id=idx,
                condition=cond_typed,
                condition_label=CONDITION_LABELS.get(cond_typed, cond),
                mitigated=is_mitigated,
                defense_label=(
                    "With Countermeasure (Anchor + HITL)"
                    if is_mitigated
                    else "Without Countermeasure (Naive)"
                ),
                window=win_id,
                window_label=WINDOW_LABELS.get(win_id, f"W{win_id}"),
                displacement_mae_m=mae,
                tail_95_error_m=tail_95,
                data_discarded_pct=discarded,
                decisions_deferred_pct=deferred,
                recovery_gain_pct=recovery_gain_pct,
                model_compute_latency_ms=latency_ms,
            )
        )

    return MatrixSummaryResponse(
        window=window,
        window_label=WINDOW_LABELS.get(window, f"W{window}"),
        avg_human_review_s=avg_review_s,
        runs=runs,
    )


@app.post("/api/hitl/override", response_model=HitlOverrideResponse)
def post_hitl_override(payload: HitlOverrideRequest) -> HitlOverrideResponse:
    init_and_seed_db()
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = get_connection()
    try:
        next_id_row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM hitl_audit_log"
        ).fetchone()
        next_id = int(next_id_row[0]) if next_id_row else 1

        lat_row = conn.execute(
            """
            SELECT model_compute_latency_ms
            FROM experiment_metrics
            WHERE condition = ? AND window_id = ? AND mitigated = TRUE
            LIMIT 1
            """,
            (payload.condition, payload.window),
        ).fetchone()
        model_latency_ms = float(lat_row[0]) if lat_row else 13.4

        vessel_row = conn.execute(
            """
            SELECT vessel_name
            FROM vessel_trajectories
            WHERE mmsi = ?
            LIMIT 1
            """,
            (payload.mmsi,),
        ).fetchone()
        vessel_name = str(vessel_row[0]) if vessel_row else "MV VIRGINIA TRADER"

        action_label = (
            "Deferred to Tier-1 USCG Coastal Anchor Kinematic Extrapolation"
            if payload.action == "DEFER_TO_ANCHOR_KINEMATIC"
            else "Logged Watch Officer Manual Course Override"
        )

        conn.execute(
            """
            INSERT INTO hitl_audit_log (
                id,
                timestamp,
                mmsi,
                vessel_name,
                condition,
                window_id,
                action_taken,
                review_duration_seconds,
                operator_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                next_id,
                now_dt.replace(tzinfo=None),
                payload.mmsi,
                vessel_name,
                payload.condition,
                payload.window,
                payload.action,
                round(payload.review_duration_seconds, 2),
                action_label,
            ),
        )

        stats_row = conn.execute(
            "SELECT COUNT(*), COALESCE(AVG(review_duration_seconds), 0.0) FROM hitl_audit_log"
        ).fetchone()
        total_interventions = int(stats_row[0]) if stats_row else 1
        avg_review_s = round(float(stats_row[1]), 2) if stats_row else round(payload.review_duration_seconds, 2)
    finally:
        conn.close()

    entry = HitlOverrideAuditEntry(
        audit_id=next_id,
        mmsi=payload.mmsi,
        action=payload.action,
        review_duration_seconds=round(payload.review_duration_seconds, 2),
        condition=payload.condition,
        window=payload.window,
        logged_at_utc=now_iso,
        model_compute_latency_ms=model_latency_ms,
    )

    return HitlOverrideResponse(
        status="RECORDED",
        message=(
            f"{action_label} for MMSI {payload.mmsi}. "
            f"Human verification effort ({entry.review_duration_seconds:.1f}s) logged separately "
            f"from model compute latency ({model_latency_ms:.1f}ms)."
        ),
        audit_entry=entry,
        total_interventions_logged=total_interventions,
        avg_human_review_s=avg_review_s,
    )
