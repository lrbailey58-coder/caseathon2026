export type ConditionType = 'NOMINAL' | 'STALE' | 'LOW_QUALITY' | 'COMBINED'
export type VesselPresetType = 'CARGO_CONTAINER' | 'HARBOR_TUG_RARE_MANEUVER'

export type LatLon = [number, number]

export interface WindowMetrics {
  displacement_mae_m: number
  tail_95_error_m: number
  data_discarded_pct: number
  decisions_deferred_pct: number
  model_compute_latency_ms: number
  avg_human_review_s: number
}

export interface TimelinePoint {
  window: string
  unmitigated_mae: number
  mitigated_mae: number
  tier2_corrupted_volume_pct: number
  avg_data_staleness_min: number
}

export interface HitlAlertPayload {
  vessel_name: string
  mmsi: string
  vessel_type?: VesselPresetType
  tier1_vs_tier2_divergence_m: number
  staleness_minutes: number
  trust_score: number
  requires_human_review: boolean
  explanation: string
  cog_delta_deg_per_min?: number
  false_positive_gate_flag?: boolean
}

export interface ScenarioData {
  condition: ConditionType
  mitigated: boolean
  window: number
  window_label: string
  vessel_type?: VesselPresetType
  ground_truth_track: LatLon[]
  unmitigated_prediction_track: LatLon[]
  mitigated_prediction_track: LatLon[]
  uncertainty_polygon: LatLon[]
  metrics: WindowMetrics
  timeline_series: TimelinePoint[]
  hitl_alert: HitlAlertPayload
}

export interface HitlOverrideRequest {
  mmsi: string
  action: 'DEFER_TO_ANCHOR_KINEMATIC' | 'MANUAL_COURSE_OVERRIDE'
  review_duration_seconds: number
  condition: ConditionType
  window: number
  operator_notes?: string
}

export interface HitlOverrideResponse {
  status: string
  message: string
  audit_entry: {
    audit_id: number
    mmsi: string
    vessel_name?: string
    action: string
    review_duration_seconds: number
    condition: ConditionType
    window: number
    logged_at_utc: string
    model_compute_latency_ms: number
    operator_notes?: string
  }
  total_interventions_logged: number
  avg_human_review_s: number
}

export interface HitlAuditLogItem {
  id: number
  timestamp: string
  mmsi: string
  vessel_name: string
  condition: ConditionType
  window_id: number
  action_taken: string
  review_duration_seconds: number
  operator_notes: string
  model_compute_latency_ms: number
}

export interface HitlLogsData {
  total_interventions: number
  avg_human_verification_s: number
  model_compute_latency_ms: number
  logs: HitlAuditLogItem[]
}

export interface MatrixRunEntry {
  run_id: number
  condition: ConditionType
  condition_label: string
  mitigated: boolean
  defense_label: string
  window: number
  window_label: string
  displacement_mae_m: number
  tail_95_error_m: number
  data_discarded_pct: number
  decisions_deferred_pct: number
  recovery_gain_pct: number
  model_compute_latency_ms: number
}

export interface MatrixSummaryData {
  window: number
  window_label: string
  avg_human_review_s: number
  runs: MatrixRunEntry[]
}
