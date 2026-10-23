import {
  Activity,
  AlertTriangle,
  Anchor,
  Clock,
  Download,
  Play,
  ShieldAlert,
  ShieldCheck,
  Ship,
  Table2,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { ConditionType, VesselPresetType } from '@/types/maritime'

export type StoryboardStepId = 1 | 2 | 3 | 4

interface HeaderControlsProps {
  condition: ConditionType
  onConditionChange: (c: ConditionType) => void
  mitigated: boolean
  onMitigatedChange: (m: boolean) => void
  window: number
  onWindowChange: (w: number) => void
  vesselType: VesselPresetType
  onVesselTypeChange: (v: VesselPresetType) => void
  windowLabel: string
  apiOnline: boolean
  onOpenMatrix: () => void
  onSelectStoryboardStep: (step: StoryboardStepId) => void
  onExportResultsJson: () => void
  exportingJson: boolean
}

const CONDITIONS: { value: ConditionType; label: string; short: string }[] = [
  { value: 'NOMINAL', label: 'Reference / Nominal', short: 'Clean Feed' },
  { value: 'STALE', label: 'Stale Only', short: 'Cached Timestamp Lag' },
  { value: 'LOW_QUALITY', label: 'Low-Quality Only', short: 'GPS Jitter & SOG Spikes' },
  { value: 'COMBINED', label: 'Combined Poisoning', short: 'Stale + Multipath Drift' },
]

const WINDOWS = [
  { id: 1, tag: 'W1', desc: 'Baseline' },
  { id: 2, tag: 'W2', desc: 'Early Drift' },
  { id: 3, tag: 'W3', desc: 'Mod Drift' },
  { id: 4, tag: 'W4', desc: 'Heavy Drift' },
  { id: 5, tag: 'W5', desc: 'Severe Drift' },
  { id: 6, tag: 'W6', desc: 'Unseen Test Period' },
]

export function HeaderControls({
  condition,
  onConditionChange,
  mitigated,
  onMitigatedChange,
  window,
  onWindowChange,
  vesselType,
  onVesselTypeChange,
  windowLabel,
  apiOnline,
  onOpenMatrix,
  onSelectStoryboardStep,
  onExportResultsJson,
  exportingJson,
}: HeaderControlsProps) {
  const isRareManeuver = vesselType === 'HARBOR_TUG_RARE_MANEUVER'

  // Identify which storyboard step matches the active state
  const activeStep: StoryboardStepId | null = isRareManeuver
    ? 4
    : condition === 'NOMINAL' && window === 1 && !mitigated
      ? 1
      : condition === 'COMBINED' && window === 5 && !mitigated
        ? 2
        : condition === 'COMBINED' && window === 6 && mitigated
          ? 3
          : null

  return (
    <header className="w-full border-b border-slate-800 bg-slate-900/95 px-4 py-2 shadow-md z-20 shrink-0 space-y-2">
      {/* Row 1: Brand Title + Controls + Export / Matrix Buttons */}
      <div className="flex flex-wrap items-center justify-between gap-2.5">
        {/* Left: Brand Title & API Status */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-500/40 bg-cyan-500/10 text-cyan-400 shadow-inner">
            <Anchor className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-sm font-bold tracking-wider text-slate-100 uppercase font-mono">
                NEPTUNE-GUARD <span className="text-cyan-400">//</span> MARITIME AIS DRIFT &amp; OVERSIGHT CONSOLE
              </h1>
              <Badge variant={apiOnline ? 'emerald' : 'amber'} className="gap-1">
                <Activity className="h-3 w-3" />
                {apiOnline ? 'DUCKDB LIVE' : 'LOCAL SIM FALLBACK'}
              </Badge>
            </div>
            <p className="text-[11px] text-slate-400 font-mono">
              SECTOR: PORT OF VIRGINIA / CHESAPEAKE BAY APPROACH (36.98°N, 76.10°W) • HORIZON: t+15 MIN
            </p>
          </div>
        </div>

        {/* Right: Interactive Scenario Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* 1. Data Condition Selector */}
          <div className="flex flex-col bg-slate-950/90 border border-slate-800 rounded-md px-2.5 py-1">
            <label
              htmlFor="condition-select"
              className="text-[10px] font-mono uppercase tracking-wider text-slate-400"
            >
              1. Stream Condition
            </label>
            <select
              id="condition-select"
              value={condition}
              onChange={(e) => onConditionChange(e.target.value as ConditionType)}
              className="bg-transparent text-xs font-semibold text-slate-100 focus:outline-none cursor-pointer pr-2 py-0.5"
            >
              {CONDITIONS.map((c) => (
                <option key={c.value} value={c.value} className="bg-slate-900 text-slate-100">
                  {c.label} ({c.short})
                </option>
              ))}
            </select>
          </div>

          {/* 2. Countermeasure Toggle */}
          <div className="flex flex-col bg-slate-950/90 border border-slate-800 rounded-md px-2.5 py-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              2. Defense Mode
            </span>
            <button
              type="button"
              onClick={() => onMitigatedChange(!mitigated)}
              className={cn(
                'flex items-center gap-1.5 rounded px-2 py-0.5 text-xs font-mono font-semibold transition-all cursor-pointer',
                mitigated
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50'
                  : 'bg-rose-500/20 text-rose-300 border border-rose-500/50',
              )}
            >
              {mitigated ? (
                <>
                  <ShieldCheck className="h-3.5 w-3.5 text-cyan-400" />
                  <span>ON: Anchor + HITL</span>
                </>
              ) : (
                <>
                  <ShieldAlert className="h-3.5 w-3.5 text-rose-400" />
                  <span>OFF: Naive Retrain</span>
                </>
              )}
            </button>
          </div>

          {/* 3. Vessel / Failure Mode Preset Selector */}
          <div
            className={cn(
              'flex flex-col rounded-md px-2.5 py-1 border transition-colors',
              isRareManeuver
                ? 'bg-amber-950/40 border-amber-500/60'
                : 'bg-slate-950/90 border-slate-800',
            )}
          >
            <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1">
              {isRareManeuver ? (
                <AlertTriangle className="h-3 w-3 text-amber-400" />
              ) : (
                <Ship className="h-3 w-3 text-cyan-400" />
              )}
              3. Vessel / Failure Mode
            </span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => onVesselTypeChange('CARGO_CONTAINER')}
                className={cn(
                  'rounded px-2 py-0.5 text-[11px] font-mono font-semibold transition-colors cursor-pointer',
                  !isRareManeuver
                    ? 'bg-cyan-500 text-slate-950'
                    : 'bg-slate-800/80 text-slate-400 hover:text-slate-200',
                )}
              >
                Cargo Transit
              </button>
              <button
                type="button"
                onClick={() => onVesselTypeChange('HARBOR_TUG_RARE_MANEUVER')}
                title="Failure Mode: TUG CAPE HENRY (MMSI 367004890) executing a high-G emergency escort turn (>45 deg/min)"
                className={cn(
                  'rounded px-2 py-0.5 text-[11px] font-mono font-semibold transition-colors cursor-pointer flex items-center gap-1',
                  isRareManeuver
                    ? 'bg-amber-500 text-slate-950 shadow-sm'
                    : 'bg-amber-500/15 text-amber-300 border border-amber-500/40 hover:bg-amber-500/25',
                )}
              >
                <span>Failure Mode: Tug Turn</span>
              </button>
            </div>
          </div>

          {/* 4. Rolling Time Window Selector */}
          <div className="flex flex-col bg-slate-950/90 border border-slate-800 rounded-md px-2.5 py-1 min-w-[225px]">
            <div className="flex items-center justify-between gap-2 mb-0.5">
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1">
                <Clock className="h-3 w-3 text-cyan-400" />
                4. Window:
              </span>
              <span className="text-[11px] font-mono font-semibold text-cyan-300">
                {windowLabel}
              </span>
            </div>
            <div className="flex items-center gap-1">
              {WINDOWS.map((w) => (
                <button
                  key={w.id}
                  type="button"
                  onClick={() => onWindowChange(w.id)}
                  title={`${w.tag}: ${w.desc}`}
                  className={cn(
                    'flex-1 rounded py-0.5 text-[11px] font-mono font-semibold transition-colors cursor-pointer',
                    window === w.id
                      ? 'bg-cyan-500 text-slate-950 shadow-sm'
                      : w.id === 6
                        ? 'bg-amber-500/15 text-amber-300 border border-amber-500/40 hover:bg-amber-500/25'
                        : 'bg-slate-800/80 text-slate-400 hover:bg-slate-700 hover:text-slate-200',
                  )}
                >
                  {w.tag}
                </button>
              ))}
            </div>
          </div>

          {/* 5. Full 4x2 Experiment Matrix Modal Button */}
          <button
            type="button"
            onClick={onOpenMatrix}
            className="flex items-center gap-1.5 rounded-md border border-cyan-500/50 bg-cyan-500/15 hover:bg-cyan-500/25 px-2.5 py-2 text-xs font-mono font-semibold text-cyan-300 shadow-sm transition-colors cursor-pointer"
          >
            <Table2 className="h-4 w-4 text-cyan-400" />
            <span>4x2 Matrix</span>
          </button>

          {/* 6. Export Results JSON Button */}
          <button
            type="button"
            disabled={exportingJson}
            onClick={onExportResultsJson}
            className="flex items-center gap-1.5 rounded-md border border-emerald-500/50 bg-emerald-500/15 hover:bg-emerald-500/25 px-2.5 py-2 text-xs font-mono font-semibold text-emerald-300 shadow-sm transition-colors cursor-pointer disabled:opacity-50"
            title="Download machine-readable JSON bundle with 8-condition metrics, manifest metadata, modification log, and HITL latency stats"
          >
            <Download className="h-4 w-4 text-emerald-400" />
            <span>{exportingJson ? 'Exporting...' : 'Export Results JSON'}</span>
          </button>
        </div>
      </div>

      {/* Row 2: 1-Click Live Presentation Storyboard Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-slate-800/90 bg-slate-950/90 px-3 py-1.5">
        <div className="flex items-center gap-1.5 text-[11px] font-mono font-bold uppercase tracking-wider text-cyan-300">
          <Play className="h-3.5 w-3.5 text-cyan-400 fill-cyan-400/20" />
          <span>Live Presentation Storyboard:</span>
        </div>

        <div className="flex flex-wrap items-center gap-1.5 flex-1 justify-end">
          <button
            type="button"
            onClick={() => onSelectStoryboardStep(1)}
            className={cn(
              'rounded px-2.5 py-1 text-[11px] font-mono font-semibold transition-all cursor-pointer border',
              activeStep === 1
                ? 'bg-emerald-500 text-slate-950 border-emerald-400 shadow-sm'
                : 'bg-slate-900/90 text-slate-300 border-slate-800 hover:border-emerald-500/50 hover:text-emerald-300',
            )}
          >
            [1. Baseline Reference (W1)]
          </button>

          <button
            type="button"
            onClick={() => onSelectStoryboardStep(2)}
            className={cn(
              'rounded px-2.5 py-1 text-[11px] font-mono font-semibold transition-all cursor-pointer border',
              activeStep === 2
                ? 'bg-rose-500 text-white border-rose-400 shadow-sm'
                : 'bg-slate-900/90 text-slate-300 border-slate-800 hover:border-rose-500/50 hover:text-rose-300',
            )}
          >
            [2. Unmitigated Drift (W5 Combined)]
          </button>

          <button
            type="button"
            onClick={() => onSelectStoryboardStep(3)}
            className={cn(
              'rounded px-2.5 py-1 text-[11px] font-mono font-semibold transition-all cursor-pointer border',
              activeStep === 3
                ? 'bg-cyan-500 text-slate-950 border-cyan-400 shadow-sm'
                : 'bg-slate-900/90 text-slate-300 border-slate-800 hover:border-cyan-500/50 hover:text-cyan-300',
            )}
          >
            [3. Countermeasure Recovery (W6 Unseen)]
          </button>

          <button
            type="button"
            onClick={() => onSelectStoryboardStep(4)}
            className={cn(
              'rounded px-2.5 py-1 text-[11px] font-mono font-semibold transition-all cursor-pointer border',
              activeStep === 4
                ? 'bg-amber-500 text-slate-950 border-amber-400 shadow-sm'
                : 'bg-slate-900/90 text-amber-300 border-amber-500/40 hover:bg-amber-500/20',
            )}
          >
            [4. Boundary Failure Case (Tugboat Maneuver)]
          </button>
        </div>
      </div>
    </header>
  )
}
