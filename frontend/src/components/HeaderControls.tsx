import { Activity, Anchor, Clock, ShieldAlert, ShieldCheck } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { ConditionType } from '@/types/maritime'

interface HeaderControlsProps {
  condition: ConditionType
  onConditionChange: (c: ConditionType) => void
  mitigated: boolean
  onMitigatedChange: (m: boolean) => void
  window: number
  onWindowChange: (w: number) => void
  windowLabel: string
  apiOnline: boolean
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
  { id: 6, tag: 'W6', desc: 'Unseen Test' },
]

export function HeaderControls({
  condition,
  onConditionChange,
  mitigated,
  onMitigatedChange,
  window,
  onWindowChange,
  windowLabel,
  apiOnline,
}: HeaderControlsProps) {
  return (
    <header className="w-full border-b border-slate-800 bg-slate-900/95 px-4 py-2.5 shadow-md z-20 shrink-0">
      <div className="flex flex-wrap items-center justify-between gap-3">
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
                {apiOnline ? 'API LIVE' : 'LOCAL SIM FALLBACK'}
              </Badge>
            </div>
            <p className="text-[11px] text-slate-400 font-mono">
              SECTOR: PORT OF VIRGINIA / CHESAPEAKE BAY APPROACH (36.98°N, 76.10°W) • HORIZON: t+15 MIN
            </p>
          </div>
        </div>

        {/* Right: Interactive Scenario Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* 1. Data Condition Selector */}
          <div className="flex flex-col bg-slate-950/90 border border-slate-800 rounded-md px-2.5 py-1">
            <label
              htmlFor="condition-select"
              className="text-[10px] font-mono uppercase tracking-wider text-slate-400"
            >
              1. Training Stream Condition
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
                  <span>ON: Anchor-Verified + HITL</span>
                </>
              ) : (
                <>
                  <ShieldAlert className="h-3.5 w-3.5 text-rose-400" />
                  <span>OFF: Naive Retraining</span>
                </>
              )}
            </button>
          </div>

          {/* 3. Rolling Time Window Selector / Slider */}
          <div className="flex flex-col bg-slate-950/90 border border-slate-800 rounded-md px-3 py-1 min-w-[260px]">
            <div className="flex items-center justify-between gap-2 mb-0.5">
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1">
                <Clock className="h-3 w-3 text-cyan-400" />
                3. Rolling Window:
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
                      : 'bg-slate-800/80 text-slate-400 hover:bg-slate-700 hover:text-slate-200',
                  )}
                >
                  {w.tag}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
