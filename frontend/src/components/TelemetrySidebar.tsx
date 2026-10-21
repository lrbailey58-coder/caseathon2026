import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Cpu,
  Gauge,
  ShieldAlert,
  TrendingUp,
  UserCheck,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { HitlOverrideRequest, HitlOverrideResponse, ScenarioData } from '@/types/maritime'

interface TelemetrySidebarProps {
  scenario: ScenarioData
  onSubmitOverride: (
    action: HitlOverrideRequest['action'],
    reviewSeconds: number,
  ) => Promise<HitlOverrideResponse>
}

export function TelemetrySidebar({ scenario, onSubmitOverride }: TelemetrySidebarProps) {
  const { metrics, timeline_series, hitl_alert, condition, window: windowNum, mitigated } = scenario

  // Live Watch Officer inspection stopwatch (resets when scenario selection changes)
  const [reviewSeconds, setReviewSeconds] = useState<number>(0)
  const [submitting, setSubmitting] = useState<boolean>(false)
  const [lastAuditToast, setLastAuditToast] = useState<HitlOverrideResponse | null>(null)

  useEffect(() => {
    const start = Date.now()
    const timer = globalThis.setInterval(() => {
      setReviewSeconds(Number(((Date.now() - start) / 1000).toFixed(1)))
    }, 100)
    return () => globalThis.clearInterval(timer)
  }, [condition, windowNum, mitigated])

  const handleOverrideClick = async (action: HitlOverrideRequest['action']) => {
    setSubmitting(true)
    try {
      const res = await onSubmitOverride(action, reviewSeconds)
      setLastAuditToast(res)
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
            <Badge variant={mitigated ? 'default' : 'rose'}>
              {mitigated ? 'Mitigated' : 'Unmitigated'}
            </Badge>
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
                  metrics.displacement_mae_m > 150
                    ? 'text-rose-400'
                    : metrics.displacement_mae_m > 75
                      ? 'text-amber-300'
                      : 'text-emerald-400'
                }`}
              >
                {metrics.displacement_mae_m.toFixed(1)} <span className="text-xs">m</span>
              </div>
              <div className="text-[10px] text-slate-500 font-mono">t+15m Horizon Error</div>
            </div>

            {/* 95th Percentile Tail Error */}
            <div className="rounded-md border border-slate-800 bg-slate-950/80 p-2.5">
              <div className="text-[10px] font-mono uppercase text-slate-400">
                95th Percentile Tail Error
              </div>
              <div
                className={`mt-0.5 text-lg font-mono font-bold ${
                  metrics.tail_95_error_m > 300 ? 'text-rose-400' : 'text-cyan-300'
                }`}
              >
                {metrics.tail_95_error_m.toFixed(0)} <span className="text-xs">m</span>
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
              <div className="text-[10px] text-slate-500 font-mono">Anchor Gate Rejected</div>
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
                  Human Review Effort
                </div>
                <div className="text-sm font-mono font-bold text-amber-300">
                  {metrics.avg_human_review_s.toFixed(1)} s avg
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
        </CardHeader>
        <CardContent className="p-2.5">
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={timeline_series}
                margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
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
                />
                <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace' }} />
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
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="mitigated_mae"
                  name="Anchor-Mitigated MAE (m)"
                  stroke="#22d3ee"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* CARD 3: Watch Officer HITL Oversight & Override Panel */}
      <Card
        className={
          hitl_alert.requires_human_review
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
          </div>
        </CardHeader>
        <CardContent className="p-3 space-y-2.5">
          {/* Vessel Alert Header + Live Stopwatch */}
          <div className="flex items-center justify-between rounded border border-slate-800 bg-slate-950 px-2.5 py-1.5">
            <div>
              <div className="text-xs font-mono font-bold text-slate-100">
                {hitl_alert.vessel_name} ({hitl_alert.mmsi})
              </div>
              <div className="text-[11px] font-mono text-slate-400">
                Divergence: <span className="text-rose-400 font-semibold">{hitl_alert.tier1_vs_tier2_divergence_m}m</span> • Feed Age:{' '}
                <span className="text-amber-300 font-semibold">{hitl_alert.staleness_minutes} min</span>
              </div>
            </div>

            <div className="flex items-center gap-1.5 rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-amber-300 font-mono text-xs">
              <Clock className="h-3.5 w-3.5 animate-pulse" />
              <span>Review Timer: {reviewSeconds.toFixed(1)}s</span>
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

          {/* Watch Officer Action Buttons */}
          <div className="space-y-1.5 pt-1">
            <Button
              variant="default"
              className="w-full justify-center font-mono text-xs"
              disabled={submitting}
              onClick={() => handleOverrideClick('DEFER_TO_ANCHOR_KINEMATIC')}
            >
              [Defer to Tier-1 Coastal Anchor Kinematic Extrapolation]
            </Button>
            <Button
              variant="amber"
              className="w-full justify-center font-mono text-xs"
              disabled={submitting}
              onClick={() => handleOverrideClick('MANUAL_COURSE_OVERRIDE')}
            >
              [Log Manual Course Override]
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
        </CardContent>
      </Card>
    </aside>
  )
}
