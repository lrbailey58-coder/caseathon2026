import { useEffect, useState } from 'react'
import { ExperimentMatrixModal } from '@/components/ExperimentMatrixModal'
import { HeaderControls } from '@/components/HeaderControls'
import { MaritimeMap } from '@/components/MaritimeMap'
import { TelemetrySidebar } from '@/components/TelemetrySidebar'
import { buildFallbackMatrixSummary, buildFallbackScenario } from '@/lib/fallbackData'
import type {
  ConditionType,
  HitlOverrideRequest,
  HitlOverrideResponse,
  MatrixSummaryData,
  ScenarioData,
} from '@/types/maritime'

export function App() {
  const [condition, setCondition] = useState<ConditionType>('COMBINED')
  const [mitigated, setMitigated] = useState<boolean>(true)
  const [window, setWindow] = useState<number>(5)
  const [apiOnline, setApiOnline] = useState<boolean>(false)
  const [scenario, setScenario] = useState<ScenarioData>(() =>
    buildFallbackScenario('COMBINED', true, 5),
  )

  const [matrixOpen, setMatrixOpen] = useState<boolean>(false)
  const [matrixData, setMatrixData] = useState<MatrixSummaryData>(() =>
    buildFallbackMatrixSummary(5),
  )

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    async function fetchScenario() {
      try {
        const params = new URLSearchParams({
          condition,
          mitigated: String(mitigated),
          window: String(window),
        })
        const res = await fetch(`/api/scenario?${params.toString()}`, {
          signal: controller.signal,
        })
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = (await res.json()) as ScenarioData
        if (!cancelled) {
          setScenario(data)
          setApiOnline(true)
        }
      } catch {
        if (!cancelled) {
          setScenario(buildFallbackScenario(condition, mitigated, window))
          setApiOnline(false)
        }
      }
    }

    fetchScenario()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [condition, mitigated, window])

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    async function fetchMatrixSummary() {
      try {
        const params = new URLSearchParams({ window: String(window) })
        const res = await fetch(`/api/matrix-summary?${params.toString()}`, {
          signal: controller.signal,
        })
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = (await res.json()) as MatrixSummaryData
        if (!cancelled) {
          setMatrixData(data)
        }
      } catch {
        if (!cancelled) {
          setMatrixData(buildFallbackMatrixSummary(window))
        }
      }
    }

    fetchMatrixSummary()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [window])

  const handleOverrideSubmit = async (
    action: HitlOverrideRequest['action'],
    reviewSeconds: number,
  ): Promise<HitlOverrideResponse> => {
    const payload: HitlOverrideRequest = {
      mmsi: scenario.hitl_alert.mmsi,
      action,
      review_duration_seconds: reviewSeconds,
      condition,
      window,
    }

    try {
      const res = await fetch('/api/hitl/override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = (await res.json()) as HitlOverrideResponse
      setScenario((prev) => ({
        ...prev,
        metrics: {
          ...prev.metrics,
          avg_human_review_s: data.avg_human_review_s,
        },
      }))
      setMatrixData((prev) => ({
        ...prev,
        avg_human_review_s: data.avg_human_review_s,
      }))
      return data
    } catch {
      const fallbackEntry: HitlOverrideResponse = {
        status: 'RECORDED_LOCAL',
        message: `${
          action === 'DEFER_TO_ANCHOR_KINEMATIC'
            ? 'Deferred to Tier-1 USCG Coastal Anchor Kinematic Extrapolation'
            : 'Logged Watch Officer Manual Course Override'
        } for MMSI ${payload.mmsi}. Human verification effort (${reviewSeconds.toFixed(1)}s) logged separately from model compute latency (${scenario.metrics.model_compute_latency_ms.toFixed(1)}ms).`,
        audit_entry: {
          audit_id: Math.floor(Math.random() * 900 + 100),
          mmsi: payload.mmsi,
          action: payload.action,
          review_duration_seconds: reviewSeconds,
          condition,
          window,
          logged_at_utc: new Date().toISOString(),
          model_compute_latency_ms: scenario.metrics.model_compute_latency_ms,
        },
        total_interventions_logged: 1,
        avg_human_review_s: reviewSeconds,
      }
      setScenario((prev) => ({
        ...prev,
        metrics: {
          ...prev.metrics,
          avg_human_review_s: reviewSeconds,
        },
      }))
      setMatrixData((prev) => ({
        ...prev,
        avg_human_review_s: reviewSeconds,
      }))
      return fallbackEntry
    }
  }

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-slate-950 text-slate-100">
      <HeaderControls
        condition={condition}
        onConditionChange={setCondition}
        mitigated={mitigated}
        onMitigatedChange={setMitigated}
        window={window}
        onWindowChange={setWindow}
        windowLabel={scenario.window_label}
        apiOnline={apiOnline}
        onOpenMatrix={() => setMatrixOpen(true)}
      />

      <main className="flex-1 flex overflow-hidden">
        <section className="w-[65%] h-full relative">
          <MaritimeMap scenario={scenario} />
        </section>
        <section className="w-[35%] h-full">
          <TelemetrySidebar scenario={scenario} onSubmitOverride={handleOverrideSubmit} />
        </section>
      </main>

      <ExperimentMatrixModal
        open={matrixOpen}
        onClose={() => setMatrixOpen(false)}
        matrixData={matrixData}
        selectedWindow={window}
        onWindowChange={setWindow}
        activeCondition={condition}
        activeMitigated={mitigated}
        onSelectRun={(c, m, w) => {
          setCondition(c)
          setMitigated(m)
          setWindow(w)
        }}
      />
    </div>
  )
}

export default App
