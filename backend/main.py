from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.db import (
    CONDITION_LABELS,
    DATA_SOURCE,
    DB_PATH,
    REPO_ROOT,
    WINDOW_LABELS,
    ConditionType,
    VesselPresetType,
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
    vessel_type: VesselPresetType = "CARGO_CONTAINER"
    tier1_vs_tier2_divergence_m: float
    staleness_minutes: float
    trust_score: float
    requires_human_review: bool
    explanation: str
    cog_delta_deg_per_min: float = 4.2
    false_positive_gate_flag: bool = False
    clean_false_alarm_pct: float | None = None  # real data: % of this ship's valid reports the gate flags on clean data


class ScenarioResponse(BaseModel):
    condition: ConditionType
    mitigated: bool
    window: int
    window_label: str
    vessel_type: VesselPresetType = "CARGO_CONTAINER"
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
    operator_notes: str | None = Field(default=None)


class HitlOverrideAuditEntry(HitlOverrideRequest):
    audit_id: int
    vessel_name: str = "MV VIRGINIA TRADER"
    logged_at_utc: str
    model_compute_latency_ms: float
    operator_notes: str = ""


class HitlOverrideResponse(BaseModel):
    status: str
    message: str
    audit_entry: HitlOverrideAuditEntry
    total_interventions_logged: int
    avg_human_review_s: float


class HitlAuditLogItem(BaseModel):
    id: int
    timestamp: str
    mmsi: str
    vessel_name: str
    condition: ConditionType
    window_id: int
    action_taken: str
    review_duration_seconds: float
    operator_notes: str
    model_compute_latency_ms: float


class HitlLogsResponse(BaseModel):
    total_interventions: int
    avg_human_verification_s: float
    model_compute_latency_ms: float
    logs: list[HitlAuditLogItem]


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
        "data_source": DATA_SOURCE,
    }


@app.get("/api/scenario", response_model=ScenarioResponse)
def get_scenario(
    condition: ConditionType = Query("COMBINED", description="Data degradation condition"),
    mitigated: bool = Query(True, description="Whether Anchor-Verified Trust Gate + HITL is active"),
    window: int = Query(5, ge=1, le=6, description="Rolling retraining window (1=W1 .. 6=W6)"),
    vessel_type: VesselPresetType = Query(
        "CARGO_CONTAINER",
        description="Vessel scenario preset: CARGO_CONTAINER (MV VIRGINIA TRADER) or HARBOR_TUG_RARE_MANEUVER (TUG CAPE HENRY - Req. 7 boundary case)",
    ),
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
                trust_score,
                example_mae_m,
                example_tail_95_m,
                example_discarded_pct,
                example_deferred_pct,
                max_turn_deg_per_min,
                clean_false_alarm_pct,
                explanation
            FROM vessel_trajectories
            WHERE condition = ? AND mitigated = ? AND window_id = ? AND vessel_type = ?
            LIMIT 1
            """,
            (condition, mitigated, window, vessel_type),
        ).fetchone()

        if not traj_row:
            raise HTTPException(
                status_code=404,
                detail=f"No vessel_trajectories found for condition={condition}, mitigated={mitigated}, window={window}, vessel_type={vessel_type}",
            )

        mmsi = str(traj_row[0])
        vessel_name = str(traj_row[1])
        gt_track = _parse_latlon_json(str(traj_row[2]))
        unmit_track = _parse_latlon_json(str(traj_row[3]))
        mit_track = _parse_latlon_json(str(traj_row[4]))
        cone_poly = _parse_latlon_json(str(traj_row[5]))
        if not cone_poly:
            # When mitigated=False in real export, uncertainty_poly_json is [].
            # Fetch the calibrated 90% uncertainty polygon from the mitigated row so the map
            # can always display the uncertainty boundary / operator-verified corridor.
            mit_cone_row = conn.execute(
                """
                SELECT uncertainty_poly_json
                FROM vessel_trajectories
                WHERE condition = ? AND mitigated = TRUE AND window_id = ? AND vessel_type = ?
                LIMIT 1
                """,
                (condition, window, vessel_type),
            ).fetchone()
            if mit_cone_row and mit_cone_row[0]:
                cone_poly = _parse_latlon_json(str(mit_cone_row[0]))
        divergence_m = float(traj_row[6])
        traj_staleness_min = float(traj_row[7])
        trust_score = float(traj_row[8])
        # Real-export fields for this example ship (all None when the demo seed data is showing)
        example_mae, example_tail_95, example_discarded, example_deferred = traj_row[9:13]
        max_turn_deg_per_min, clean_false_alarm_pct, real_explanation = traj_row[13:16]
        has_real_example = example_mae is not None

        # Adjust local scenario metrics when inspecting the Req. 7 Harbor Tug rare maneuver boundary case
        if vessel_type == "HARBOR_TUG_RARE_MANEUVER" and has_real_example:
            # Real data: this tug's own error, flagged share and hand-off share from the ML export
            mae = float(example_mae)
            tail_95 = float(example_tail_95)
            discarded_pct = float(example_discarded)
            deferred_pct = float(example_deferred)
        elif vessel_type == "HARBOR_TUG_RARE_MANEUVER":  # demo values (no real results file)
            if mitigated:
                # Adverse tradeoff: Trust Gate discards 82% of valid high-frequency turn updates, causing turn lag
                mae = round(268.0 + (window - 1) * 14.5, 1)
                tail_95 = round(495.0 + (window - 1) * 28.0, 1)
                discarded_pct = 82.4
                deferred_pct = 100.0
            else:
                # Unmitigated model retains raw high-frequency Tier-2 turn pings (lower lag on clean/low drift, worse under heavy poison)
                mae = round(92.0 + (0.0 if condition == "NOMINAL" else (window - 1) * 44.0), 1)
                tail_95 = round(185.0 + (0.0 if condition == "NOMINAL" else (window - 1) * 86.0), 1)
                discarded_pct = 0.0
                deferred_pct = 0.0

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

    is_tug_maneuver = vessel_type == "HARBOR_TUG_RARE_MANEUVER"
    requires_review = is_tug_maneuver or (condition != "NOMINAL" and window >= 2)
    if real_explanation:
        explanation = str(real_explanation)
    else:
        explanation = build_explanation(
            condition=condition,
            window_id=window,
            staleness_min=staleness_min,
            corrupted_pct=corrupted_pct,
            divergence_m=divergence_m,
            vessel_type=vessel_type,
            discarded_pct=discarded_pct,
        )
    if has_real_example:
        cog_delta = float(max_turn_deg_per_min)
        false_positive_flag = is_tug_maneuver and float(clean_false_alarm_pct) > 0
    else:  # demo values
        cog_delta = 64.2 if is_tug_maneuver else 4.2
        false_positive_flag = is_tug_maneuver

    return ScenarioResponse(
        condition=condition,
        mitigated=mitigated,
        window=window,
        window_label=WINDOW_LABELS.get(window, f"W{window}"),
        vessel_type=vessel_type,
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
            vessel_type=vessel_type,
            tier1_vs_tier2_divergence_m=divergence_m,
            staleness_minutes=traj_staleness_min,
            trust_score=trust_score,
            requires_human_review=requires_review,
            explanation=explanation,
            cog_delta_deg_per_min=cog_delta,
            false_positive_gate_flag=false_positive_flag,
            clean_false_alarm_pct=float(clean_false_alarm_pct) if has_real_example else None,
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

    # Clean data without the countermeasure = the reference for "damage undone"
    clean_unmit_mae = next((float(r[3]) for r in rows if str(r[0]) == "NOMINAL" and not bool(r[1])), None)

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

        # Recovery = share of the poisoning damage the countermeasure undid:
        #   (error without countermeasure - error with it) / (error without countermeasure - clean error)
        # For clean data there is no damage, so it shows the countermeasure's cost instead
        # (relative change vs. no countermeasure; negative = slightly worse).
        damage = unmit_mae - clean_unmit_mae if clean_unmit_mae is not None else 0.0
        if is_mitigated and cond != "NOMINAL" and damage > 1e-9:
            recovery_gain_pct = round((unmit_mae - mae) / damage * 100.0, 1)
        elif is_mitigated and unmit_mae > 0:
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
            SELECT vessel_name, vessel_type, max_turn_deg_per_min
            FROM vessel_trajectories
            WHERE mmsi = ?
            LIMIT 1
            """,
            (payload.mmsi,),
        ).fetchone()
        vessel_name = str(vessel_row[0]) if vessel_row else (
            "TUG CAPE HENRY" if payload.mmsi == "367004890" else "MV VIRGINIA TRADER"
        )
        is_tug_vessel = (
            (vessel_row is not None and str(vessel_row[1]) == "HARBOR_TUG_RARE_MANEUVER")
            or payload.mmsi == "367004890"
        )
        max_turn_val = float(vessel_row[2]) if (vessel_row and vessel_row[2] is not None) else 64.2

        default_action_note = (
            "Deferred to Tier-1 USCG Coastal Anchor Kinematic Extrapolation"
            if payload.action == "DEFER_TO_ANCHOR_KINEMATIC"
            else (
                f"Watch Officer Manual Waypoint Override: Verified legitimate sharp harbor tug turn (COG delta = {max_turn_val:.0f} deg/min)"
                if is_tug_vessel
                else "Logged Watch Officer Manual Waypoint / Course Override"
            )
        )
        operator_notes = (
            payload.operator_notes.strip()
            if payload.operator_notes and payload.operator_notes.strip()
            else default_action_note
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
                operator_notes,
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
        vessel_name=vessel_name,
        action=payload.action,
        review_duration_seconds=round(payload.review_duration_seconds, 2),
        condition=payload.condition,
        window=payload.window,
        logged_at_utc=now_iso,
        model_compute_latency_ms=model_latency_ms,
        operator_notes=operator_notes,
    )

    return HitlOverrideResponse(
        status="RECORDED",
        message=(
            f"{operator_notes} for {vessel_name} (MMSI {payload.mmsi}). "
            f"Human verification effort ({entry.review_duration_seconds:.1f}s) logged separately "
            f"from model compute latency ({model_latency_ms:.1f}ms)."
        ),
        audit_entry=entry,
        total_interventions_logged=total_interventions,
        avg_human_review_s=avg_review_s,
    )


@app.get("/api/hitl/logs", response_model=HitlLogsResponse)
def get_hitl_logs(
    limit: int = Query(25, ge=1, le=100, description="Maximum number of recent HITL audit log entries to return"),
) -> HitlLogsResponse:
    init_and_seed_db()
    conn = get_connection()
    try:
        stats_row = conn.execute(
            "SELECT COUNT(*), COALESCE(AVG(review_duration_seconds), 0.0) FROM hitl_audit_log"
        ).fetchone()
        total_interventions = int(stats_row[0]) if stats_row else 0
        avg_verification_s = round(float(stats_row[1]), 2) if stats_row else 0.0

        lat_row = conn.execute(
            "SELECT COALESCE(AVG(model_compute_latency_ms), 13.4) FROM experiment_metrics WHERE mitigated = TRUE"
        ).fetchone()
        avg_model_latency_ms = round(float(lat_row[0]), 2) if lat_row else 13.4

        rows = conn.execute(
            """
            SELECT
                h.id,
                h.timestamp,
                h.mmsi,
                h.vessel_name,
                h.condition,
                h.window_id,
                h.action_taken,
                h.review_duration_seconds,
                h.operator_notes,
                COALESCE(e.model_compute_latency_ms, 13.4) AS model_compute_latency_ms
            FROM hitl_audit_log h
            LEFT JOIN experiment_metrics e
              ON h.condition = e.condition
             AND h.window_id = e.window_id
             AND e.mitigated = TRUE
            ORDER BY h.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    logs: list[HitlAuditLogItem] = []
    for r in rows:
        raw_ts = r[1]
        if isinstance(raw_ts, datetime):
            ts_iso = raw_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            ts_iso = str(raw_ts)

        cond_val: ConditionType = str(r[4])  # type: ignore[assignment]
        logs.append(
            HitlAuditLogItem(
                id=int(r[0]),
                timestamp=ts_iso,
                mmsi=str(r[2]),
                vessel_name=str(r[3]),
                condition=cond_val,
                window_id=int(r[5]),
                action_taken=str(r[6]),
                review_duration_seconds=round(float(r[7]), 2),
                operator_notes=str(r[8]),
                model_compute_latency_ms=round(float(r[9]), 1),
            )
        )

    return HitlLogsResponse(
        total_interventions=total_interventions,
        avg_human_verification_s=avg_verification_s,
        model_compute_latency_ms=avg_model_latency_ms,
        logs=logs,
    )


def _failure_mode_summary() -> dict[str, object]:
    """Requirement 7 boundary case: the real W6 tug example when real results are loaded, else the demo text."""
    if DATA_SOURCE.get("kind") == "real":
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT mitigated, vessel_name, mmsi, voyage_id, max_turn_deg_per_min, clean_false_alarm_pct,
                       example_mae_m, example_discarded_pct, example_deferred_pct, explanation
                FROM vessel_trajectories
                WHERE vessel_type = 'HARBOR_TUG_RARE_MANEUVER' AND window_id = 6 AND condition = 'NOMINAL'
                ORDER BY mitigated
                """
            ).fetchall()
        finally:
            conn.close()
        if len(rows) == 2:
            off, on = rows
            return {
                "preset_id": "HARBOR_TUG_RARE_MANEUVER",
                "source": "real AIS voyage (W6, clean data, so every gate flag is a false alarm)",
                "vessel": f"{on[1]} (MMSI {on[2]}, voyage {on[3]})",
                "max_turn_deg_per_min": on[4],
                "clean_false_alarm_pct": on[5],
                "example_error_gate_off_m": off[6],
                "example_error_gate_on_m": on[6],
                "share_of_its_fast_reports_flagged_pct": on[7],
                "share_of_its_predictions_handed_to_human_pct": on[8],
                "adverse_tradeoff": on[9],
                "proposed_next_step": (
                    "Compare fast-feed reports against a curved path (anchor positions plus their headings) "
                    "instead of a straight line between anchors, then re-test sharp-turn false alarms."
                ),
            }
    return {
        "preset_id": "HARBOR_TUG_RARE_MANEUVER",
        "source": "demo values (no real results file loaded)",
        "vessel": "TUG CAPE HENRY (MMSI 367004890)",
        "maneuver": "High-G legitimate emergency escort turn in Norfolk channel (COG delta = 64.2 deg/min > 45 deg/min threshold)",
        "adverse_tradeoff": (
            "Kinematic + Tier-1 Anchor Trust Gate falsely flags 82.4% of valid high-frequency Tier-2 turn updates "
            "as Low-Quality / Sensor Spikes, causing the mitigated model to lag along the pre-turn anchor tangent "
            "until Watch Officer manual waypoint verification."
        ),
    }


@app.get("/api/export/results")
def export_results_bundle() -> dict[str, object]:
    """Export a clean machine-readable JSON bundle containing the 8-condition metrics table,
    dataset manifest metadata, modification log summary, and HITL latency stats."""
    init_and_seed_db()

    # 1. 8-run experimental matrix on unseen evaluation window W6 (plus all windows W1..W6)
    w6_matrix = get_matrix_summary(window=6)
    hitl_summary = get_hitl_logs(limit=25)

    conn = get_connection()
    try:
        all_metrics_rows = conn.execute(
            """
            SELECT
                condition,
                mitigated,
                window_id,
                displacement_mae_m,
                tail_95_error_m,
                data_discarded_pct,
                decisions_deferred_pct,
                tier2_corrupted_volume_pct,
                avg_data_staleness_min,
                model_compute_latency_ms
            FROM experiment_metrics
            ORDER BY window_id ASC, condition ASC, mitigated ASC
            """
        ).fetchall()
    finally:
        conn.close()

    rolling_metrics_all_windows = [
        {
            "condition": str(r[0]),
            "mitigated": bool(r[1]),
            "window_id": int(r[2]),
            "window_label": WINDOW_LABELS.get(int(r[2]), f"W{int(r[2])}"),
            "displacement_mae_m": float(r[3]),
            "tail_95_error_m": float(r[4]),
            "data_discarded_pct": float(r[5]),
            "decisions_deferred_pct": float(r[6]),
            "tier2_corrupted_volume_pct": float(r[7]),
            "avg_data_staleness_min": float(r[8]),
            "model_compute_latency_ms": float(r[9]),
        }
        for r in all_metrics_rows
    ]

    # 2. Load dataset manifest metadata
    manifest_path = REPO_ROOT / "backend" / "database" / "manifest.json"
    dataset_manifest: dict[str, object] = {}
    if manifest_path.exists():
        try:
            dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            dataset_manifest = {"error": "Unable to parse manifest.json"}

    # 3. Load modification log summary
    mod_log_path = REPO_ROOT / "backend" / "database" / "modification_log.json"
    modification_log_summary: dict[str, object] = {}
    if mod_log_path.exists():
        try:
            modification_log_summary = json.loads(mod_log_path.read_text(encoding="utf-8"))
        except Exception:
            modification_log_summary = {"error": "Unable to parse modification_log.json"}

    return {
        "schema_version": "1.0.0",
        "exported_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": "NEPTUNE-GUARD // W&M AI Case-a-thon 2026 (Unintentional Data Poisoning)",
        "evaluation_unit": "Complete chronological vessel voyages (15-minute ahead prediction horizon)",
        "primary_metrics": ["displacement_mae_m", "tail_95_error_m"],
        "eight_condition_metrics_table_w6": [run.model_dump() for run in w6_matrix.runs],
        "rolling_window_metrics_w1_to_w6": rolling_metrics_all_windows,
        "dataset_manifest_metadata": dataset_manifest,
        "modification_log_summary": modification_log_summary,
        "hitl_oversight_and_latency_stats": {
            "total_interventions": hitl_summary.total_interventions,
            "avg_human_verification_seconds": hitl_summary.avg_human_verification_s,
            "avg_model_compute_latency_ms": hitl_summary.model_compute_latency_ms,
            "latency_separation_note": (
                "Model compute latency (milliseconds) is tracked and reported strictly separately "
                "from Watch Officer human verification/override duration (seconds)."
            ),
            "recent_audit_trail": [item.model_dump() for item in hitl_summary.logs],
        },
        "data_source": DATA_SOURCE,
        "failure_mode_boundary_analysis": _failure_mode_summary(),
    }


FRONTEND_DIST_DIR = Path("frontend/dist")
if not FRONTEND_DIST_DIR.exists():
    FRONTEND_DIST_DIR = REPO_ROOT / "frontend" / "dist"

if FRONTEND_DIST_DIR.exists():
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="frontend-assets",
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_react_spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"API route not found: /{full_path}")
        candidate = (FRONTEND_DIST_DIR / full_path).resolve()
        if (
            full_path
            and candidate.is_file()
            and FRONTEND_DIST_DIR.resolve() in candidate.parents
        ):
            return FileResponse(str(candidate))
        return FileResponse("frontend/dist/index.html" if Path("frontend/dist/index.html").exists() else str(FRONTEND_DIST_DIR / "index.html"))

