from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import duckdb
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="NEPTUNE-GUARD // Maritime AIS Drift & Oversight API",
    version="0.1.0",
    description="Day 1 Tactical API for W&M Case-a-thon 2026: Unintentional Data Poisoning",
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

ConditionType = Literal["NOMINAL", "STALE", "LOW_QUALITY", "COMBINED"]

# Base ground-truth voyage along Chesapeake Bay / Thimble Shoal Channel into Port of Virginia (Norfolk)
BASE_GROUND_TRUTH_TRACK: list[tuple[float, float]] = [
    (36.9520, -75.9850),
    (36.9610, -76.0120),
    (36.9705, -76.0410),
    (36.9790, -76.0695),
    (36.9865, -76.0980),
    (36.9935, -76.1260),
    (36.9995, -76.1540),
    (37.0040, -76.1810),
    (37.0065, -76.2080),
    (37.0045, -76.2350),
    (36.9985, -76.2610),
    (36.9890, -76.2860),
]

# Deterministic progression tables across rolling windows W1 -> W6
ERROR_PROGRESSION: dict[ConditionType, dict[str, list[float]]] = {
    "NOMINAL": {
        "unmitigated_mae": [42.0, 44.5, 43.8, 45.2, 46.1, 47.0],
        "mitigated_mae": [41.5, 42.8, 42.1, 43.0, 43.6, 44.2],
        "tail_95_unmitigated": [88.0, 92.0, 91.0, 94.5, 96.0, 98.0],
        "tail_95_mitigated": [85.0, 88.0, 87.5, 89.0, 90.5, 91.5],
        "corrupted_vol_pct": [2.0, 2.5, 2.2, 2.8, 3.0, 3.1],
        "staleness_min": [1.2, 1.3, 1.2, 1.4, 1.4, 1.5],
        "discarded_pct": [1.8, 2.1, 2.0, 2.4, 2.6, 2.8],
        "deferred_pct": [2.0, 2.0, 2.5, 2.5, 3.0, 3.0],
    },
    "STALE": {
        "unmitigated_mae": [43.0, 118.0, 215.0, 342.0, 468.0, 540.0],
        "mitigated_mae": [42.0, 56.0, 68.0, 79.0, 88.0, 94.0],
        "tail_95_unmitigated": [90.0, 245.0, 430.0, 685.0, 910.0, 1060.0],
        "tail_95_mitigated": [86.0, 115.0, 138.0, 162.0, 179.0, 192.0],
        "corrupted_vol_pct": [4.0, 18.0, 34.0, 51.0, 66.0, 72.0],
        "staleness_min": [1.5, 6.8, 12.4, 17.5, 22.8, 25.4],
        "discarded_pct": [3.2, 16.5, 31.0, 47.5, 61.2, 67.8],
        "deferred_pct": [2.5, 6.0, 10.5, 14.0, 18.5, 21.0],
    },
    "LOW_QUALITY": {
        "unmitigated_mae": [44.0, 134.0, 248.0, 389.0, 512.0, 595.0],
        "mitigated_mae": [42.5, 59.0, 74.0, 86.0, 96.0, 105.0],
        "tail_95_unmitigated": [92.0, 290.0, 520.0, 790.0, 1045.0, 1210.0],
        "tail_95_mitigated": [88.0, 124.0, 152.0, 178.0, 198.0, 215.0],
        "corrupted_vol_pct": [5.0, 20.0, 38.0, 55.0, 68.0, 75.0],
        "staleness_min": [1.4, 2.1, 2.8, 3.4, 3.9, 4.2],
        "discarded_pct": [4.1, 18.2, 35.4, 51.0, 63.8, 70.2],
        "deferred_pct": [3.0, 7.5, 12.0, 16.5, 20.0, 23.5],
    },
    "COMBINED": {
        "unmitigated_mae": [45.0, 165.0, 310.0, 495.0, 670.0, 785.0],
        "mitigated_mae": [43.0, 64.0, 82.0, 98.0, 112.0, 124.0],
        "tail_95_unmitigated": [95.0, 355.0, 640.0, 980.0, 1320.0, 1540.0],
        "tail_95_mitigated": [89.0, 134.0, 170.0, 204.0, 230.0, 252.0],
        "corrupted_vol_pct": [6.0, 24.0, 44.0, 62.0, 76.0, 82.0],
        "staleness_min": [1.6, 8.2, 14.8, 19.6, 24.5, 27.8],
        "discarded_pct": [5.0, 21.8, 40.5, 57.4, 71.0, 76.5],
        "deferred_pct": [3.5, 9.0, 15.0, 21.5, 26.0, 29.5],
    },
}


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


# In-memory HITL intervention store for Day 1 demo
HITL_AUDIT_LOG: list[HitlOverrideAuditEntry] = []


def _build_tracks(
    condition: ConditionType, window: int, mitigated: bool
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    """Generate deterministic ground truth, unmitigated drift, mitigated recovery, and 95% cone."""
    w_idx = window - 1  # 0..5
    drift_scale = 0.0 if condition == "NOMINAL" else (w_idx / 5.0)

    unmitigated_track: list[tuple[float, float]] = []
    mitigated_track: list[tuple[float, float]] = []
    upper_bound: list[tuple[float, float]] = []
    lower_bound: list[tuple[float, float]] = []

    for i, (lat, lon) in enumerate(BASE_GROUND_TRUTH_TRACK):
        progress = max(0.0, (i - 2) / (len(BASE_GROUND_TRUTH_TRACK) - 3))
        if i <= 2:
            unmitigated_track.append((lat, lon))
            mitigated_track.append((lat, lon))
            continue

        # Unmitigated drift pattern depends on poisoning mode
        if condition == "NOMINAL":
            d_lat = 0.0005 * progress
            d_lon = 0.0004 * progress
        elif condition == "STALE":
            # Stale positions lag behind actual trajectory and pull vessel north-east out of the turn
            d_lat = 0.0145 * drift_scale * (progress**1.35)
            d_lon = 0.0180 * drift_scale * (progress**1.25)
        elif condition == "LOW_QUALITY":
            # Multipath jitter + SOG/COG spikes cause erratic southward shoal excursion
            osc = 0.0035 if i % 2 == 0 else -0.0022
            d_lat = (-0.0155 * (progress**1.3) + osc) * drift_scale
            d_lon = 0.0110 * drift_scale * progress
        else:  # COMBINED
            osc = 0.0030 if i % 2 == 0 else -0.0015
            d_lat = (0.0195 * (progress**1.35) + osc) * drift_scale
            d_lon = 0.0225 * drift_scale * (progress**1.25)

        unmit_lat = round(lat + d_lat, 6)
        unmit_lon = round(lon + d_lon, 6)
        unmitigated_track.append((unmit_lat, unmit_lon))

        # Anchor-Verified Trust Gate recovers trajectory within tight channel bounds
        mit_lat = round(lat + d_lat * 0.14, 6)
        mit_lon = round(lon + d_lon * 0.14, 6)
        mitigated_track.append((mit_lat, mit_lon))

        # 95% Quantile Uncertainty Cone around active prediction track
        center_lat = mit_lat if mitigated else unmit_lat
        center_lon = mit_lon if mitigated else unmit_lon
        cone_half_width = (0.0014 + 0.0022 * drift_scale) * (0.45 + 0.85 * progress)
        if not mitigated and condition != "NOMINAL":
            cone_half_width *= 1.65

        upper_bound.append((round(center_lat + cone_half_width, 6), round(center_lon, 6)))
        lower_bound.append((round(center_lat - cone_half_width, 6), round(center_lon, 6)))

    anchor_start = BASE_GROUND_TRUTH_TRACK[2]
    uncertainty_polygon = [anchor_start] + upper_bound + list(reversed(lower_bound))

    return BASE_GROUND_TRUTH_TRACK, unmitigated_track, mitigated_track, uncertainty_polygon


@app.get("/api/health")
def get_health() -> dict[str, object]:
    duckdb_ready = False
    duckdb_version = "unknown"
    try:
        conn = duckdb.connect(":memory:")
        row = conn.execute("SELECT version()").fetchone()
        if row:
            duckdb_version = str(row[0])
        duckdb_ready = True
        conn.close()
    except Exception:
        duckdb_ready = False

    return {
        "status": "ONLINE",
        "service": "neptune-guard-api",
        "duckdb_ready": duckdb_ready,
        "duckdb_version": duckdb_version,
        "region": "Port of Virginia / Chesapeake Bay Approach",
        "logged_hitl_interventions": len(HITL_AUDIT_LOG),
    }


@app.get("/api/scenario", response_model=ScenarioResponse)
def get_scenario(
    condition: ConditionType = Query("COMBINED", description="Data degradation condition"),
    mitigated: bool = Query(True, description="Whether Anchor-Verified Trust Gate + HITL is active"),
    window: int = Query(5, ge=1, le=6, description="Rolling retraining window (1=W1 .. 6=W6)"),
) -> ScenarioResponse:
    w_idx = window - 1
    table = ERROR_PROGRESSION[condition]

    gt_track, unmit_track, mit_track, cone_poly = _build_tracks(condition, window, mitigated)

    mae = table["mitigated_mae"][w_idx] if mitigated else table["unmitigated_mae"][w_idx]
    tail_95 = (
        table["tail_95_mitigated"][w_idx]
        if mitigated
        else table["tail_95_unmitigated"][w_idx]
    )
    discarded_pct = table["discarded_pct"][w_idx] if mitigated else 0.0
    deferred_pct = table["deferred_pct"][w_idx] if mitigated else 0.0
    model_latency_ms = round(11.8 + (1.6 if mitigated else 0.0) + 0.2 * w_idx, 1)

    if HITL_AUDIT_LOG:
        avg_review_s = round(
            sum(item.review_duration_seconds for item in HITL_AUDIT_LOG) / len(HITL_AUDIT_LOG),
            1,
        )
    else:
        avg_review_s = 8.4 if mitigated else 0.0

    timeline_series = [
        TimelinePoint(
            window=f"W{i + 1}",
            unmitigated_mae=table["unmitigated_mae"][i],
            mitigated_mae=table["mitigated_mae"][i],
            tier2_corrupted_volume_pct=table["corrupted_vol_pct"][i],
            avg_data_staleness_min=table["staleness_min"][i],
        )
        for i in range(6)
    ]

    staleness_min = table["staleness_min"][w_idx]
    corrupted_pct = table["corrupted_vol_pct"][w_idx]
    divergence_m = round(table["unmitigated_mae"][w_idx] * 0.88, 1)
    trust_score = round(max(0.12, 0.98 - (corrupted_pct / 100.0) * 0.85), 2)
    requires_review = condition != "NOMINAL" and window >= 2

    if condition == "NOMINAL":
        explanation = (
            "Tier-2 commercial stream aligns within 45m of delayed Tier-1 USCG coastal anchor. "
            "Kinematic velocity and heading deltas remain within nominal Norfolk channel bounds."
        )
    elif condition == "STALE":
        explanation = (
            f"Tier-2 Fast Feed repeats cached vessel coordinates (avg age {staleness_min:.1f} min) "
            f"with fresh timestamps, deviating {divergence_m:.0f}m from Tier-1 USCG Anchor trajectory."
        )
    elif condition == "LOW_QUALITY":
        explanation = (
            f"Tier-2 Fast Feed exhibits severe GPS multipath jitter and SOG/COG spikes across "
            f"{corrupted_pct:.0f}% of retraining buffer; deviates {divergence_m:.0f}m from Tier-1 Anchor."
        )
    else:
        explanation = (
            f"Combined poisoning: Tier-2 Fast Feed has {staleness_min:.1f} min staleness and "
            f"{corrupted_pct:.0f}% corrupted records, pulling unmitigated t+15 prediction {divergence_m:.0f}m "
            "outside Thimble Shoal inbound channel."
        )

    window_labels = {
        1: "W1: Baseline Reference",
        2: "W2: Early Feed Drift",
        3: "W3: Moderate Poisoning",
        4: "W4: Heavy Retraining Drift",
        5: "W5: Severe Channel Excursion",
        6: "W6: Unseen Evaluation Period",
    }

    return ScenarioResponse(
        condition=condition,
        mitigated=mitigated,
        window=window,
        window_label=window_labels[window],
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
            vessel_name="MV VIRGINIA TRADER",
            mmsi="367591240",
            tier1_vs_tier2_divergence_m=divergence_m,
            staleness_minutes=staleness_min,
            trust_score=trust_score,
            requires_human_review=requires_review,
            explanation=explanation,
        ),
    )


@app.post("/api/hitl/override", response_model=HitlOverrideResponse)
def post_hitl_override(payload: HitlOverrideRequest) -> HitlOverrideResponse:
    model_latency_ms = 13.2
    entry = HitlOverrideAuditEntry(
        audit_id=len(HITL_AUDIT_LOG) + 1,
        mmsi=payload.mmsi,
        action=payload.action,
        review_duration_seconds=round(payload.review_duration_seconds, 2),
        condition=payload.condition,
        window=payload.window,
        logged_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_compute_latency_ms=model_latency_ms,
    )
    HITL_AUDIT_LOG.append(entry)

    avg_review_s = round(
        sum(item.review_duration_seconds for item in HITL_AUDIT_LOG) / len(HITL_AUDIT_LOG),
        2,
    )

    action_label = (
        "Deferred to Tier-1 USCG Coastal Anchor Kinematic Extrapolation"
        if payload.action == "DEFER_TO_ANCHOR_KINEMATIC"
        else "Logged Watch Officer Manual Course Override"
    )

    return HitlOverrideResponse(
        status="RECORDED",
        message=(
            f"{action_label} for MMSI {payload.mmsi}. "
            f"Human verification effort ({entry.review_duration_seconds:.1f}s) logged separately "
            f"from model compute latency ({model_latency_ms:.1f}ms)."
        ),
        audit_entry=entry,
        total_interventions_logged=len(HITL_AUDIT_LOG),
        avg_human_review_s=avg_review_s,
    )
