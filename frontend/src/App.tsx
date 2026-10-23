import { useEffect, useState } from 'react'
import { ExperimentMatrixModal } from '@/components/ExperimentMatrixModal'
import { HeaderControls, type StoryboardStepId } from '@/components/HeaderControls'
import { MaritimeMap } from '@/components/MaritimeMap'
import { TelemetrySidebar } from '@/components/TelemetrySidebar'
import {
  buildFallbackExportBundle,
  buildFallbackHitlLogs,
  buildFallbackMatrixSummary,
  buildFallbackScenario,
} from '@/lib/fallbackData'
import type {
  ConditionType,
  HitlLogsData,
  HitlOverrideRequest,
  HitlOverrideResponse,
  MatrixSummaryData,
  ScenarioData,
  VesselPresetType,
} from '@/types/maritime'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') || ''

export function App() {
  const [condition, setCondition] = useState<ConditionType>('COMBINED')
  const [mitigated, setMitigated] = useState<boolean>(true)
  const [window, setWindow] = useState<number>(5)
  const [vesselType, setVesselType] = useState<VesselPresetType>('CARGO_CONTAINER')
  const [apiOnline, setApiOnline] = useState<boolean>(false)
  const [exportingJson, setExportingJson] = useState<boolean>(false)
  const [scenario, setScenario] = useState<ScenarioData>(() =>
    buildFallbackScenario('COMBINED', true, 5, 'CARGO_CONTAINER'),
  )
  const [operatorVerified, setOperatorVerified] = useState<{
    key: string
    action: HitlOverrideRequest['action']
  } | null>(null)
  const [hitlLogs, setHitlLogs] = useState<HitlLogsData>(() => buildFallbackHitlLogs())

  const [matrixOpen, setMatrixOpen] = useState<boolean>(false)
  const [matrixData, setMatrixData] = useState<MatrixSummaryData>(() =>
    buildFallbackMatrixSummary(5),
  )

  const currentScenarioKey = `${condition}-${String(mitigated)}-${window}-${vesselType}`
  const operatorVerifiedAction =
    operatorVerified?.key === currentScenarioKey ? operatorVerified.action : null

  const handleSelectStoryboardStep = (step: StoryboardStepId) => {
    if (step === 1) {
      // [1. Baseline Reference (W1)] -> Loads NOMINAL, W1, Countermeasure OFF
      setCondition('NOMINAL')
      setWindow(1)
      setMitigated(false)
      setVesselType('CARGO_CONTAINER')
    } else if (step === 2) {
      // [2. Unmitigated Drift (W5 Combined)] -> Loads COMBINED, W5, Countermeasure OFF
      setCondition('COMBINED')
      setWindow(5)
      setMitigated(false)
      setVesselType('CARGO_CONTAINER')
    } else if (step === 3) {
      // [3. Countermeasure Recovery (W6 Unseen)] -> Loads COMBINED, W6, Countermeasure ON
      setCondition('COMBINED')
      setWindow(6)
      setMitigated(true)
      setVesselType('CARGO_CONTAINER')
    } else {
      // [4. Boundary Failure Case (Tugboat Maneuver)] -> Loads Harbor Tug false-positive tradeoff scenario
      setCondition('COMBINED')
      setWindow(5)
      setMitigated(true)
      setVesselType('HARBOR_TUG_RARE_MANEUVER')
    }
  }

  const handleExportResultsJson = async () => {
    setExportingJson(true)
    try {
      let bundle: Record<string, unknown>
      try {
        const res = await fetch(`${API_BASE_URL}/api/export/results`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        bundle = (await res.json()) as Record<string, unknown>
      } catch {
        bundle = buildFallbackExportBundle(hitlLogs)
      }

      const jsonString = JSON.stringify(bundle, null, 2)
      const blob = new Blob([jsonString], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `neptune-guard-results-manifest-w${window}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
    } finally {
      setExportingJson(false)
    }
  }

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    async function fetchScenario() {
      try {
        const params = new URLSearchParams({
          condition,
          mitigated: String(mitigated),
          window: String(window),
          vessel_type: vesselType,
        })
        const res = await fetch(`${API_BASE_URL}/api/scenario?${params.toString()}`, {
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
          setScenario(buildFallbackScenario(condition, mitigated, window, vesselType))
          setApiOnline(false)
        }
      }
    }

    fetchScenario()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [condition, mitigated, window, vesselType])

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    async function fetchHitlLogs() {
      try {
        const res = await fetch(`${API_BASE_URL}/api/hitl/logs?limit=25`, {
          signal: controller.signal,
        })
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = (await res.json()) as HitlLogsData
        if (!cancelled) {
          setHitlLogs(data)
        }
      } catch {
        // Keep existing local audit log state when offline
      }
    }

    fetchHitlLogs()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [condition, window, vesselType])

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    async function fetchMatrixSummary() {
      try {
        const params = new URLSearchParams({ window: String(window) })
        const res = await fetch(`${API_BASE_URL}/api/matrix-summary?${params.toString()}`, {
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
    operatorNotes?: string,
  ): Promise<HitlOverrideResponse> => {
    const payload: HitlOverrideRequest = {
      mmsi: scenario.hitl_alert.mmsi,
      action,
      review_duration_seconds: reviewSeconds,
      condition,
      window,
      operator_notes: operatorNotes,
    }

    // Immediately animate/update active trajectory on the map and badge vessel marker as OPERATOR VERIFIED
    setOperatorVerified({ key: currentScenarioKey, action })

    try {
      const res = await fetch(`${API_BASE_URL}/api/hitl/override`, {
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

      // Refresh persisted DuckDB logs from GET /api/hitl/logs
      try {
        const logsRes = await fetch(`${API_BASE_URL}/api/hitl/logs?limit=25`)
        if (logsRes.ok) {
          const logsData = (await logsRes.json()) as HitlLogsData
          setHitlLogs(logsData)
        }
      } catch {
        // Ignore secondary fetch error
      }

      return data
    } catch {
      const defaultNote =
        operatorNotes ||
        (action === 'DEFER_TO_ANCHOR_KINEMATIC'
          ? 'Deferred to Tier-1 USCG Coastal Anchor Kinematic Extrapolation'
          : 'Logged Watch Officer Manual Waypoint Override')

      const nextId = (hitlLogs.logs[0]?.id ?? 0) + 1
      const newTotal = hitlLogs.total_interventions + 1
      const newAvg = Number(
        (
          (hitlLogs.avg_human_verification_s * hitlLogs.total_interventions + reviewSeconds) /
          newTotal
        ).toFixed(2),
      )

      const fallbackEntry: HitlOverrideResponse = {
        status: 'RECORDED_LOCAL',
        message: `${defaultNote} for ${scenario.hitl_alert.vessel_name} (MMSI ${payload.mmsi}). Human verification effort (${reviewSeconds.toFixed(1)}s) logged separately from model compute latency (${scenario.metrics.model_compute_latency_ms.toFixed(1)}ms).`,
        audit_entry: {
          audit_id: nextId,
          mmsi: payload.mmsi,
          vessel_name: scenario.hitl_alert.vessel_name,
          action: payload.action,
          review_duration_seconds: reviewSeconds,
          condition,
          window,
          logged_at_utc: new Date().toISOString(),
          model_compute_latency_ms: scenario.metrics.model_compute_latency_ms,
          operator_notes: defaultNote,
        },
        total_interventions_logged: newTotal,
        avg_human_review_s: newAvg,
      }

      setHitlLogs((prev) => ({
        total_interventions: newTotal,
        avg_human_verification_s: newAvg,
        model_compute_latency_ms: scenario.metrics.model_compute_latency_ms,
        logs: [
          {
            id: nextId,
            timestamp: new Date().toISOString(),
            mmsi: payload.mmsi,
            vessel_name: scenario.hitl_alert.vessel_name,
            condition,
            window_id: window,
            action_taken: payload.action,
            review_duration_seconds: reviewSeconds,
            operator_notes: defaultNote,
            model_compute_latency_ms: scenario.metrics.model_compute_latency_ms,
          },
          ...prev.logs,
        ],
      }))

      setScenario((prev) => ({
        ...prev,
        metrics: {
          ...prev.metrics,
          avg_human_review_s: newAvg,
        },
      }))
      setMatrixData((prev) => ({
        ...prev,
        avg_human_review_s: newAvg,
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
        vesselType={vesselType}
        onVesselTypeChange={setVesselType}
        windowLabel={scenario.window_label}
        apiOnline={apiOnline}
        onOpenMatrix={() => setMatrixOpen(true)}
        onSelectStoryboardStep={handleSelectStoryboardStep}
        onExportResultsJson={handleExportResultsJson}
        exportingJson={exportingJson}
      />

      <main className="flex-1 flex overflow-hidden">
        <section className="w-[65%] h-full relative">
          <MaritimeMap
            scenario={scenario}
            operatorVerifiedAction={operatorVerifiedAction}
          />
        </section>
        <section className="w-[35%] h-full">
          <TelemetrySidebar
            scenario={scenario}
            hitlLogs={hitlLogs}
            operatorVerifiedAction={operatorVerifiedAction}
            onSubmitOverride={handleOverrideSubmit}
            onSelectVesselPreset={setVesselType}
          />
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
