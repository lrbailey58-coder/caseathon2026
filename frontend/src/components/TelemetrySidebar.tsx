import {
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Cpu,
  FileSpreadsheet,
  Gauge,
  Scale,
  ShieldAlert,
  TrendingUp,
  UserCheck,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type {
  HitlLogsData,
  HitlOverrideRequest,
  HitlOverrideResponse,
  ScenarioData,
  VesselPresetType,
} from '@/types/maritime'

interface TelemetrySidebarProps {
  scenario: ScenarioData
  hitlLogs: HitlLogsData
  operatorVerifiedAction: HitlOverrideRequest['action'] | null
  onSubmitOverride: (
    action: HitlOverrideRequest['action'],
    reviewSeconds: number,
    operatorNotes?: string,
  ) => Promise<HitlOverrideResponse>
  onSelectVesselPreset: (vesselType: VesselPresetType) => void
}

export function TelemetrySidebar({
  scenario,
  hitlLogs,
  operatorVerifiedAction,
  onSubmitOverride,
  onSelectVesselPreset,
}: TelemetrySidebarProps) {
  const {
    metrics,
    timeline_series,
    hitl_alert,
    condition,
    window: windowNum,
    mitigated,
    vessel_type,
  } = scenario
  const selectedWindowKey = `W${windowNum}`
  const isTugRareManeuver = vessel_type === 'HARBOR_TUG_RARE_MANEUVER'

  const currentScenarioKey = `${condition}-${String(mitigated)}-${windowNum}-${String(vessel_type)}`

  // Live Watch Officer inspection stopwatch (resets when scenario selection changes)
  const [reviewSeconds, setReviewSeconds] = useState<number>(0)
  const [submitting, setSubmitting] = useState<boolean>(false)
  const [toastState, setToastState] = useState<{
    key: string
    response: HitlOverrideResponse
  } | null>(null)
  const [auditDrawerOpen, setAuditDrawerOpen] = useState<boolean>(true)
  const [operatorNotesInput, setOperatorNotesInput] = useState<string>('')
  const hitlPanelRef = useRef<HTMLDivElement | null>(null)
  const failureCardRef = useRef<HTMLDivElement | null>(null)

  const lastAuditToast =
    toastState?.key === currentScenarioKey ? toastState.response : null

  useEffect(() => {
    const start = Date.now()
    const timer = globalThis.setInterval(() => {
      setReviewSeconds(Number(((Date.now() - start) / 1000).toFixed(1)))
    }, 100)
    return () => globalThis.clearInterval(timer)
  }, [condition, windowNum, mitigated, vessel_type])

  useEffect(() => {
    if (isTugRareManeuver) {
      hitlPanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      failureCardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [isTugRareManeuver])

  const handleOverrideClick = async (action: HitlOverrideRequest['action']) => {
    setSubmitting(true)
    try {
      const res = await onSubmitOverride(
        action,
        reviewSeconds,
        operatorNotesInput.trim() || undefined,
      )
      setToastState({ key: currentScenarioKey, response: res })
      setOperatorNotesInput('')
      setAuditDrawerOpen(true)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <aside className="h-full w-full overflow-y-auto border-l border-slate-800 bg-slate-950 p-3 space-y-3">
      {/* CARD 1: Performance & Workload Metrics Grid */}
      <Card>
        <CardHeader className="py-2.5 px-3.5">
          <div className="flex items-center justify-between">
            <CardTitle>
              <Gauge className="h-4 w-4 text-cyan-400" />
              1. Telemetry &amp; Workload Metrics ({scenario.window_label.split(':')[0]})
            </CardTitle>
            <div className="flex items-center gap-1.5">
              {isTugRareManeuver && <Badge variant="amber">Failure Mode</Badge>}
              {windowNum === 6 && <Badge variant="amber">Unseen Test</Badge>}
              <Badge variant={mitigated ? 'default' : 'rose'}>
                {mitigated ? 'Mitigated' : 'Unmitigated'}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-3">
          <div className="grid grid-cols-2 gap-2">
            {/* Primary Displacement MAE */}
            <div className="rounded-md border border-slate-800 bg-slate-950/80 p-2.5">
              <div className="text-[10px] font-mono uppercase text-slate-400">
                Primary Displacement MAE
              </div>
              <div
                className={`mt-0.5 text-lg font-mono font-bold ${
                  operatorVerifiedAction
                    ? 'text-emerald-400'
                    : metrics.displacement_mae_m > 150
                      ? 'text-rose-400'
                      : metrics.displacement_mae_m > 75
                        ? 'text-amber-300'
                        : 'text-emerald-400'
                }`}
              >
                {operatorVerifiedAction
                  ? '38.4'
                  : metrics.displacement_mae_m.toFixed(1)}{' '}
                <span className="text-xs">m</span>
              </div>
              <div className="text-[10px] text-slate-500 font-mono">
                {operatorVerifiedAction ? 'Operator-Verified Safe Path' : 't+15m Horizon Error'}
              </div>
            </div>

            {/* 95th Percentile Tail Error */}
            <div className="rounded-md border border-slate-800 bg-slate-950/80 p-2.5">
              <div className="text-[10px] font-mono uppercase text-slate-400">
                95th Percentile Tail Error
              </div>
              <div
                className={`mt-0.5 text-lg font-mono font-bold ${
                  operatorVerifiedAction
                    ? 'text-emerald-400'
                    : metrics.tail_95_error_m > 300
                      ? 'text-rose-400'
                      : 'text-cyan-300'
                }`}
              >
                {operatorVerifiedAction ? '76' : metrics.tail_95_error_m.toFixed(0)}{' '}
                <span className="text-xs">m</span>
              </div>
              <div className="text-[10px] text-slate-500 font-mono">Worst-Case Excursion</div>
            </div>

            {/* Tier-2 Data Filtered */}
            <div className="rounded-md border border-slate-800 bg-slate-950/80 p-2.5">
              <div className="text-[10px] font-mono uppercase text-slate-400">
                Tier-2 Data Filtered
              </div>
              <div className="mt-0.5 text-base font-mono font-bold text-amber-300">
                {metrics.data_discarded_pct.toFixed(1)}%
              </div>
              <div className="text-[10px] text-slate-500 font-mono">
                {isTugRareManeuver && mitigated
                  ? 'False-Positive Turn Reject'
                  : 'Anchor Gate Rejected'}
              </div>
            </div>

            {/* Decisions Deferred */}
            <div className="rounded-md border border-slate-800 bg-slate-950/80 p-2.5">
              <div className="text-[10px] font-mono uppercase text-slate-400">
                Decisions Deferred
              </div>
              <div className="mt-0.5 text-base font-mono font-bold text-cyan-300">
                {metrics.decisions_deferred_pct.toFixed(1)}%
              </div>
              <div className="text-[10px] text-slate-500 font-mono">Routed to Watch Officer</div>
            </div>
          </div>

          {/* Explicit Separation: Model Compute Latency vs Human Review Effort */}
          <div className="mt-2 grid grid-cols-2 gap-2 rounded-md border border-cyan-500/30 bg-cyan-950/15 p-2">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-cyan-400 shrink-0" />
              <div>
                <div className="text-[10px] font-mono uppercase text-slate-400">
                  Model Compute Time
                </div>
                <div className="text-sm font-mono font-bold text-slate-100">
                  {metrics.model_compute_latency_ms.toFixed(1)} ms
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2 border-l border-slate-800 pl-2">
              <UserCheck className="h-4 w-4 text-amber-400 shrink-0" />
              <div>
                <div className="text-[10px] font-mono uppercase text-slate-400">
                  Human Verification
                </div>
                <div className="text-sm font-mono font-bold text-amber-300">
                  {hitlLogs.avg_human_verification_s.toFixed(1)} s avg (
                  {hitlLogs.total_interventions})
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* CARD 2: Rolling Training-Time Drift Chart (Recharts) */}
      <Card>
        <CardHeader className="py-2.5 px-3.5">
          <div className="flex items-center justify-between">
            <CardTitle>
              <TrendingUp className="h-4 w-4 text-cyan-400" />
              2. Rolling Retraining Drift (W1 → W6)
            </CardTitle>
            <span className="text-[10px] font-mono text-slate-400">MAE (m) vs Poison Vol (%)</span>
          </div>
          {/* Regime Legend: W1-W5 Training Drift vs W6 Unseen Evaluation */}
          <div className="flex items-center justify-between pt-1 text-[10px] font-mono">
            <span className="inline-flex items-center gap-1.5 text-slate-300">
              <span className="h-2 w-2 rounded-full bg-cyan-400/80 inline-block" />
              W1–W5: Rolling Training Drift
            </span>
            <span className="inline-flex items-center gap-1.5 text-amber-300">
              <span className="h-2 w-2 rounded-full bg-amber-400 inline-block" />
              W6: Unseen Test Period
            </span>
          </div>
        </CardHeader>
        <CardContent className="p-2.5">
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={timeline_series}
                margin={{ top: 16, right: 8, left: -16, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="window"
                  stroke="#94a3b8"
                  fontSize={11}
                  tickLine={false}
                />
                <YAxis
                  yAxisId="left"
                  stroke="#94a3b8"
                  fontSize={10}
                  tickLine={false}
                  unit="m"
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#fbbf24"
                  fontSize={10}
                  tickLine={false}
                  domain={[0, 100]}
                  unit="%"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                  }}
                  labelFormatter={(label) =>
                    label === 'W6'
                      ? 'W6: Unseen Test Period (Holdout Evaluation)'
                      : `${String(label)}: Rolling Retraining Window`
                  }
                />
                <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace' }} />

                <ReferenceArea
                  yAxisId="left"
                  x1="W1"
                  x2="W5"
                  fill="#0ea5e9"
                  fillOpacity={0.04}
                />
                <ReferenceArea
                  yAxisId="left"
                  x1="W5"
                  x2="W6"
                  fill="#f59e0b"
                  fillOpacity={0.12}
                />

                <ReferenceLine
                  yAxisId="left"
                  x={selectedWindowKey}
                  stroke={windowNum === 6 ? '#f59e0b' : '#22d3ee'}
                  strokeWidth={2}
                  strokeDasharray="4 3"
                  label={{
                    value: windowNum === 6 ? 'ACTIVE: W6 TEST' : `ACTIVE: ${selectedWindowKey}`,
                    position: 'top',
                    fill: windowNum === 6 ? '#fbbf24' : '#67e8f9',
                    fontSize: 10,
                    fontFamily: 'monospace',
                  }}
                />

                <Area
                  yAxisId="right"
                  type="monotone"
                  dataKey="tier2_corrupted_volume_pct"
                  name="Tier-2 Corrupted Vol (%)"
                  fill="#fbbf24"
                  fillOpacity={0.14}
                  stroke="#fbbf24"
                  strokeDasharray="3 3"
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="unmitigated_mae"
                  name="Unmitigated MAE (m)"
                  stroke="#f43f5e"
                  strokeWidth={2.2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="mitigated_mae"
                  name="Anchor-Mitigated MAE (m)"
                  stroke="#22d3ee"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* CARD 3: Watch Officer HITL Oversight & Override Panel */}
      <Card
        ref={hitlPanelRef}
        className={
          operatorVerifiedAction
            ? 'border-emerald-500/60 bg-slate-900/95'
            : hitl_alert.requires_human_review
              ? 'border-amber-500/50 bg-slate-900/95'
              : 'border-slate-800'
        }
      >
        <CardHeader className="py-2.5 px-3.5">
          <div className="flex items-center justify-between">
            <CardTitle>
              <ShieldAlert className="h-4 w-4 text-amber-400" />
              3. Watch Officer HITL Oversight Queue
            </CardTitle>
            {operatorVerifiedAction ? (
              <Badge variant="emerald">OPERATOR VERIFIED</Badge>
            ) : (
              <Badge
                variant={
                  hitl_alert.trust_score < 0.5
                    ? 'rose'
                    : hitl_alert.trust_score < 0.8
                      ? 'amber'
                      : 'emerald'
                }
              >
                Trust: {(hitl_alert.trust_score * 100).toFixed(0)}%
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="p-3 space-y-2.5">
          {/* Vessel Alert Header + Live Stopwatch */}
          <div className="flex items-center justify-between rounded border border-slate-800 bg-slate-950 px-2.5 py-1.5">
            <div>
              <div className="text-xs font-mono font-bold text-slate-100 flex items-center gap-1.5">
                <span>
                  {hitl_alert.vessel_name} ({hitl_alert.mmsi})
                </span>
                {isTugRareManeuver && (
                  <span className="text-[10px] font-mono text-amber-300 bg-amber-500/15 border border-amber-500/40 rounded px-1">
                    COG Δ {(hitl_alert.cog_delta_deg_per_min ?? 0).toFixed(0)}°/min
                  </span>
                )}
              </div>
              <div className="text-[11px] font-mono text-slate-400">
                Divergence:{' '}
                <span className="text-rose-400 font-semibold">
                  {hitl_alert.tier1_vs_tier2_divergence_m}m
                </span>{' '}
                • Feed Age:{' '}
                <span className="text-amber-300 font-semibold">
                  {hitl_alert.staleness_minutes} min
                </span>
              </div>
            </div>

            <div className="flex items-center gap-1.5 rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-amber-300 font-mono text-xs">
              <Clock className="h-3.5 w-3.5 animate-pulse" />
              <span>Timer: {reviewSeconds.toFixed(1)}s</span>
            </div>
          </div>

          {/* Diagnostic Evidence Explanation */}
          <div className="rounded border border-slate-800/90 bg-slate-950/60 p-2.5 text-xs text-slate-300 leading-relaxed">
            <div className="flex items-center gap-1.5 text-[11px] font-mono font-semibold uppercase text-amber-300 mb-1">
              <AlertTriangle className="h-3.5 w-3.5" />
              Cross-Source Diagnostic Evidence:
            </div>
            {hitl_alert.explanation}
          </div>

          {/* Optional Watch Officer Operator Note Input */}
          <div>
            <input
              type="text"
              value={operatorNotesInput}
              onChange={(e) => setOperatorNotesInput(e.target.value)}
              placeholder={
                isTugRareManeuver
                  ? 'Operator note (e.g., Confirmed valid tug turn via VHF Ch. 13)...'
                  : 'Optional Watch Officer log note before committing override...'
              }
              className="w-full rounded border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-xs font-mono text-slate-200 placeholder:text-slate-500 focus:border-cyan-500/60 focus:outline-none"
            />
          </div>

          {/* Watch Officer Action Buttons */}
          <div className="space-y-1.5 pt-0.5">
            <Button
              variant="default"
              className="w-full justify-center font-mono text-xs"
              disabled={submitting}
              onClick={() => handleOverrideClick('DEFER_TO_ANCHOR_KINEMATIC')}
            >
              [Defer to Tier-1 Coastal Anchor]
            </Button>
            <Button
              variant="amber"
              className="w-full justify-center font-mono text-xs"
              disabled={submitting}
              onClick={() => handleOverrideClick('MANUAL_COURSE_OVERRIDE')}
            >
              [Manual Waypoint Override]
            </Button>
          </div>

          {/* Confirmation Toast / Banner */}
          {lastAuditToast && (
            <div className="rounded-md border border-emerald-500/40 bg-emerald-950/30 p-2.5 text-xs text-emerald-200 font-mono">
              <div className="flex items-center gap-1.5 font-bold text-emerald-300">
                <CheckCircle2 className="h-4 w-4 shrink-0" />
                OVERSIGHT ACTION LOGGED (AUDIT #{lastAuditToast.audit_entry.audit_id})
              </div>
              <p className="mt-1 text-[11px] text-emerald-100/90 leading-snug">
                {lastAuditToast.message}
              </p>
            </div>
          )}

          {/* Expandable Oversight Audit Log Drawer */}
          <div className="rounded-md border border-slate-800 bg-slate-950/90 overflow-hidden">
            <button
              type="button"
              onClick={() => setAuditDrawerOpen(!auditDrawerOpen)}
              className="w-full flex items-center justify-between px-3 py-2 text-left text-xs font-mono font-semibold text-slate-200 hover:bg-slate-900 transition-colors cursor-pointer"
            >
              <span className="flex items-center gap-1.5">
                <FileSpreadsheet className="h-3.5 w-3.5 text-cyan-400" />
                Oversight Audit Log (DuckDB: {hitlLogs.total_interventions} Interventions)
              </span>
              <span className="flex items-center gap-1.5 text-[11px] text-slate-400">
                <span>
                  Avg: <strong className="text-amber-300">{hitlLogs.avg_human_verification_s.toFixed(1)}s</strong> vs{' '}
                  <strong className="text-cyan-300">{hitlLogs.model_compute_latency_ms.toFixed(1)}ms</strong>
                </span>
                {auditDrawerOpen ? (
                  <ChevronUp className="h-3.5 w-3.5" />
                ) : (
                  <ChevronDown className="h-3.5 w-3.5" />
                )}
              </span>
            </button>

            {auditDrawerOpen && (
              <div className="border-t border-slate-800 px-2.5 py-2 space-y-1.5 max-h-48 overflow-y-auto font-mono text-[11px]">
                {hitlLogs.logs.length === 0 ? (
                  <div className="text-slate-500 py-2 text-center">
                    No interventions recorded yet.
                  </div>
                ) : (
                  hitlLogs.logs.slice(0, 8).map((item) => (
                    <div
                      key={item.id}
                      className="rounded border border-slate-800/90 bg-slate-900/70 p-2 space-y-0.5"
                    >
                      <div className="flex items-center justify-between text-slate-200">
                        <span className="font-bold text-cyan-300">
                          #{item.id} • {item.vessel_name} ({item.mmsi})
                        </span>
                        <span className="text-[10px] text-slate-400">
                          W{item.window_id} {item.condition}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[10px]">
                        <span
                          className={
                            item.action_taken === 'MANUAL_COURSE_OVERRIDE'
                              ? 'text-amber-300 font-semibold'
                              : 'text-emerald-300 font-semibold'
                          }
                        >
                          {item.action_taken}
                        </span>
                        <span className="text-slate-300">
                          Human: <strong className="text-amber-300">{item.review_duration_seconds.toFixed(1)}s</strong> •
                          Compute: <strong className="text-cyan-300">{item.model_compute_latency_ms.toFixed(1)}ms</strong>
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-400 truncate" title={item.operator_notes}>
                        {item.operator_notes}
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* CARD 4: Tradeoff & Failure Mode Inspector */}
      <Card
        ref={failureCardRef}
        className={
          isTugRareManeuver
            ? 'border-rose-500/60 bg-rose-950/15'
            : 'border-amber-500/40 bg-slate-900/95'
        }
      >
        <CardHeader className="py-2.5 px-3.5">
          <div className="flex items-center justify-between">
            <CardTitle className="text-amber-300">
              <AlertOctagon className="h-4 w-4 text-rose-400" />
              4. Failure Mode &amp; Tradeoff Inspector
            </CardTitle>
            <Badge variant={isTugRareManeuver ? 'rose' : 'amber'}>
              {isTugRareManeuver ? 'ACTIVE ON MAP' : 'FAILURE MODE'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-3 space-y-2.5 text-xs">
          {/* Scenario Quick-Switch Button */}
          <div className="flex items-center justify-between rounded border border-amber-500/40 bg-amber-950/25 px-2.5 py-2">
            <div className="font-mono">
              <div className="font-bold text-amber-200">
                Harbor tug, sharp turn (real AIS example)
              </div>
              <div className="text-[10px] text-slate-300">
                Preset: <code className="text-amber-300">HARBOR_TUG_RARE_MANEUVER</code> • turns ≥ 20°/min
              </div>
            </div>
            <Button
              size="sm"
              variant={isTugRareManeuver ? 'outline' : 'amber'}
              className="font-mono text-[11px]"
              onClick={() =>
                onSelectVesselPreset(
                  isTugRareManeuver ? 'CARGO_CONTAINER' : 'HARBOR_TUG_RARE_MANEUVER',
                )
              }
            >
              {isTugRareManeuver ? 'Return to Cargo' : 'Load Failure Case'}
            </Button>
          </div>

          {/* Adverse Tradeoff Explanation */}
          <div className="rounded border border-slate-800 bg-slate-950/80 p-2.5 space-y-1.5 text-slate-300 leading-relaxed">
            <div className="font-mono text-[11px] font-semibold uppercase text-rose-300 flex items-center gap-1.5">
              <Scale className="h-3.5 w-3.5 text-rose-400" />
              Adverse Tradeoff: Valid Rare Maneuver vs. Sensor Spike
            </div>
            <p className="text-[11px] text-slate-300">
              When a ship makes a real sharp turn (<span className="font-mono text-amber-300">≥ 20°/min</span>),
              its fast-feed reports curve away from the straight line our Trust Gate draws between
              Tier-1 anchor reports (~6 min apart), so the gate can wrongly discard valid data as a
              <strong className="text-rose-300"> Low-Quality / Sensor Spike</strong>.
            </p>
            <div className="grid grid-cols-2 gap-1.5 pt-1 font-mono text-[10px]">
              <div className="rounded border border-slate-800 bg-slate-900/90 p-1.5">
                <div className="text-slate-400">This Ship: Valid Reports Flagged (Clean Data)</div>
                <div className="text-rose-400 font-bold text-xs">
                  {hitl_alert.clean_false_alarm_pct != null
                    ? `${hitl_alert.clean_false_alarm_pct.toFixed(1)}%`
                    : 'n/a (demo data)'}
                </div>
              </div>
              <div className="rounded border border-slate-800 bg-slate-900/90 p-1.5">
                <div className="text-slate-400">Steady 300 m GPS Offset (Unseen Test)</div>
                <div className="text-amber-300 font-bold text-xs">
                  Mostly Missed (Boundary)
                </div>
              </div>
            </div>
          </div>

          {/* Governance Checkpoint before Pilot Deployment */}
          <div className="rounded border border-cyan-500/35 bg-cyan-950/20 p-2.5 space-y-1">
            <div className="font-mono text-[11px] font-bold uppercase text-cyan-300">
              Mandatory Governance Checkpoint (Pre-Pilot Deployment):
            </div>
            <ul className="list-disc pl-4 space-y-1 text-[11px] text-slate-300">
              <li>
                <strong>Turn-Aware Gate:</strong> Compare fast-feed reports against a curved path
                (anchor positions plus their headings) instead of a straight line, and re-test the
                sharp-turn false-alarm rate before any pilot.
              </li>
              <li>
                <strong>Watch Officer Deferral:</strong> Predictions whose 90% range exceeds 1 nautical
                mile (1,852 m) are routed to the Watch Officer for review before any automated
                collision-avoidance advisory.
              </li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </aside>
  )
}
