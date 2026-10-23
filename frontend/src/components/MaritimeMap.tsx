import {
  AlertTriangle,
  CheckCircle2,
  Compass,
  Layers,
  Navigation,
  Radio,
  ShieldCheck,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { CircleMarker, MapContainer, Polygon, Polyline, TileLayer, Tooltip } from 'react-leaflet'
import { Badge } from '@/components/ui/badge'
import type { HitlOverrideRequest, LatLon, ScenarioData } from '@/types/maritime'

interface MaritimeMapProps {
  scenario: ScenarioData
  operatorVerifiedAction: HitlOverrideRequest['action'] | null
}

// Norfolk / Thimble Shoal Shipping Channel boundary lines for visual context
const CHANNEL_NORTH_BOUNDARY: [number, number][] = [
  [36.956, -75.983],
  [36.974, -76.039],
  [36.991, -76.096],
  [37.004, -76.153],
  [37.011, -76.208],
  [37.003, -76.261],
  [36.993, -76.286],
]

const CHANNEL_SOUTH_BOUNDARY: [number, number][] = [
  [36.948, -75.987],
  [36.966, -76.043],
  [36.982, -76.1],
  [36.995, -76.155],
  [37.002, -76.208],
  [36.994, -76.261],
  [36.984, -76.286],
]

export function MaritimeMap({ scenario, operatorVerifiedAction }: MaritimeMapProps) {
  const {
    ground_truth_track,
    unmitigated_prediction_track,
    mitigated_prediction_track,
    uncertainty_polygon,
    mitigated,
    condition,
    window,
    hitl_alert,
    vessel_type,
  } = scenario

  const isTugRareManeuver = vessel_type === 'HARBOR_TUG_RARE_MANEUVER'
  const isOperatorVerified = operatorVerifiedAction !== null

  // Compute Watch Officer corrected safe trajectory when an override is triggered
  const correctedSafeTrack: LatLon[] = ground_truth_track.map(([gtLat, gtLon], i) => {
    if (i <= 2) {
      return [gtLat, gtLon]
    }
    const progress = (i - 2) / (ground_truth_track.length - 3)
    const offsetScale = operatorVerifiedAction === 'MANUAL_COURSE_OVERRIDE' ? 0.00025 : 0.00045
    return [
      Number((gtLat + offsetScale * progress).toFixed(6)),
      Number((gtLon - offsetScale * 0.8 * progress).toFixed(6)),
    ]
  })

  // Progressive animation state when Watch Officer triggers an override
  const [animState, setAnimState] = useState<{
    action: HitlOverrideRequest['action'] | null
    count: number
  }>({ action: null, count: 3 })

  useEffect(() => {
    if (!isOperatorVerified) {
      return
    }
    const startMs = Date.now()
    const interval = globalThis.setInterval(() => {
      const steps = Math.min(
        correctedSafeTrack.length,
        3 + Math.floor((Date.now() - startMs) / 50),
      )
      setAnimState({ action: operatorVerifiedAction, count: steps })
      if (steps >= correctedSafeTrack.length) {
        globalThis.clearInterval(interval)
      }
    }, 45)
    return () => globalThis.clearInterval(interval)
  }, [isOperatorVerified, operatorVerifiedAction, correctedSafeTrack.length])

  const animatedPointCount =
    isOperatorVerified && animState.action === operatorVerifiedAction
      ? animState.count
      : 3

  const activeMitigatedTrack: LatLon[] = isOperatorVerified
    ? correctedSafeTrack.slice(0, animatedPointCount)
    : mitigated_prediction_track

  // Narrow calibrated cone around the corrected safe path after Watch Officer verification
  const activeUncertaintyPolygon: LatLon[] = isOperatorVerified
    ? (() => {
        const upper: LatLon[] = []
        const lower: LatLon[] = []
        correctedSafeTrack.forEach(([lat, lon], idx) => {
          if (idx <= 2) return
          const w = 0.00085
          upper.push([Number((lat + w).toFixed(6)), lon])
          lower.push([Number((lat - w).toFixed(6)), lon])
        })
        return [correctedSafeTrack[2], ...upper, ...lower.reverse()]
      })()
    : uncertainty_polygon

  const vesselCurrentPos = ground_truth_track[2]
  const gtEndPos = ground_truth_track[ground_truth_track.length - 1]
  const unmitEndPos = unmitigated_prediction_track[unmitigated_prediction_track.length - 1]
  const mitEndPos = activeMitigatedTrack[activeMitigatedTrack.length - 1] ?? gtEndPos

  const cartoApiKey = import.meta.env.VITE_CARTO_API_KEY?.trim()
  const cartoTileUrl = cartoApiKey
    ? `https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?key=${encodeURIComponent(cartoApiKey)}`
    : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'

  return (
    <div className="relative h-full w-full bg-slate-950 overflow-hidden select-none">
      {/* Top-Left Tactical HUD Overlay */}
      <div className="absolute top-3 left-14 z-[500] flex flex-wrap items-center gap-2 pointer-events-none">
        <div className="rounded-md border border-slate-800/90 bg-slate-900/90 px-3 py-1.5 backdrop-blur-md shadow-lg">
          <div className="flex items-center gap-2 text-xs font-mono text-slate-200">
            <Navigation className="h-3.5 w-3.5 text-cyan-400" />
            <span className="font-bold">{hitl_alert.vessel_name}</span>
            <span className="text-slate-400">MMSI {hitl_alert.mmsi}</span>
            {isTugRareManeuver && (
              <span className="rounded bg-amber-500/20 border border-amber-500/50 px-1.5 py-0.5 text-[10px] text-amber-300 font-semibold">
                FAILURE MODE: RARE ESCORT TURN
              </span>
            )}
          </div>
          <div className="mt-0.5 flex items-center gap-3 text-[11px] font-mono text-slate-400">
            <span>
              {isTugRareManeuver
                ? 'MANEUVER: HIGH-G ESCORT PIVOT (COG Δ 64.2°/min)'
                : 'CORRIDOR: THIMBLE SHOAL INBOUND'}
            </span>
            <span>•</span>
            <span>
              TRUST SCORE: {isOperatorVerified ? '100% (WATCH OFFICER)' : `${(hitl_alert.trust_score * 100).toFixed(0)}%`}
            </span>
          </div>
        </div>

        {isOperatorVerified ? (
          <Badge
            variant="emerald"
            className="px-2.5 py-1 text-xs shadow-lg backdrop-blur-md flex items-center gap-1.5 animate-pulse"
          >
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
            <span>
              OPERATOR VERIFIED • SAFE PATH LOCKED (
              {operatorVerifiedAction === 'DEFER_TO_ANCHOR_KINEMATIC'
                ? 'TIER-1 ANCHOR EXTRAPOLATION'
                : 'MANUAL WAYPOINT OVERRIDE'}
              )
            </span>
          </Badge>
        ) : isTugRareManeuver && mitigated ? (
          <Badge
            variant="amber"
            className="px-2.5 py-1 text-xs shadow-lg backdrop-blur-md flex items-center gap-1.5"
          >
            <AlertTriangle className="h-3.5 w-3.5 text-amber-300" />
            <span>
              FAILURE MODE: TRUST GATE FALSE POSITIVE • TURN LAG {scenario.metrics.displacement_mae_m.toFixed(0)}m
            </span>
          </Badge>
        ) : (
          <Badge
            variant={
              condition === 'NOMINAL'
                ? 'emerald'
                : mitigated
                  ? 'default'
                  : 'rose'
            }
            className="px-2.5 py-1 text-xs shadow-lg backdrop-blur-md"
          >
            {mitigated
              ? `ANCHOR GATE ACTIVE • MAE ${scenario.metrics.displacement_mae_m.toFixed(0)}m`
              : `UNMITIGATED DRIFT • MAE ${scenario.metrics.displacement_mae_m.toFixed(0)}m`}
          </Badge>
        )}
      </div>

      {/* Leaflet Map Canvas */}
      <MapContainer
        center={[36.968, -76.12]}
        zoom={11}
        className="h-full w-full z-0"
        zoomControl={true}
      >
        <TileLayer
          key={cartoTileUrl}
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url={cartoTileUrl}
        />

        {/* Shipping Channel Lateral Boundaries */}
        <Polyline
          positions={CHANNEL_NORTH_BOUNDARY}
          pathOptions={{
            color: '#475569',
            weight: 1.5,
            dashArray: '4 8',
            opacity: 0.65,
          }}
        />
        <Polyline
          positions={CHANNEL_SOUTH_BOUNDARY}
          pathOptions={{
            color: '#475569',
            weight: 1.5,
            dashArray: '4 8',
            opacity: 0.65,
          }}
        />

        {/* 95% Quantile Uncertainty Cone Polygon */}
        <Polygon
          positions={activeUncertaintyPolygon}
          pathOptions={{
            color: isOperatorVerified
              ? '#10b981'
              : isTugRareManeuver && mitigated
                ? '#f59e0b'
                : mitigated
                  ? '#22d3ee'
                  : '#f43f5e',
            fillColor: isOperatorVerified
              ? '#10b981'
              : isTugRareManeuver && mitigated
                ? '#f59e0b'
                : mitigated
                  ? '#22d3ee'
                  : '#f43f5e',
            fillOpacity: isOperatorVerified ? 0.18 : mitigated ? 0.16 : 0.12,
            weight: 1,
            dashArray: '3 3',
          }}
        >
          <Tooltip sticky>
            <div className="font-mono text-xs">
              {isOperatorVerified
                ? 'Operator-Verified Safe Corridor (Watch Officer Locked)'
                : `95% Quantile Prediction Cone (${mitigated ? 'Anchor-Calibrated' : 'Unmitigated Spread'})`}
            </div>
          </Tooltip>
        </Polygon>

        {/* Ground Truth Track (Emerald Solid Line) */}
        <Polyline
          positions={ground_truth_track}
          pathOptions={{
            color: '#34d399',
            weight: 3.5,
            opacity: 0.95,
          }}
        >
          <Tooltip sticky>
            <span className="font-mono text-xs">
              Ground-Truth Vessel Trajectory ({hitl_alert.vessel_name})
            </span>
          </Tooltip>
        </Polyline>

        {/* Unmitigated Poisoned Prediction Track (Rose Dashed Line) */}
        <Polyline
          positions={unmitigated_prediction_track}
          pathOptions={{
            color: '#f43f5e',
            weight: mitigated && !isOperatorVerified ? 2.5 : isOperatorVerified ? 2 : 4,
            dashArray: '8 6',
            opacity: isOperatorVerified ? 0.35 : mitigated ? 0.65 : 0.95,
          }}
        >
          <Tooltip sticky>
            <span className="font-mono text-xs">
              Unmitigated Model Prediction (W{window} {condition})
            </span>
          </Tooltip>
        </Polyline>

        {/* When Operator Verified on Tugboat or Poisoned scenario, show ghosted pre-override track for contrast */}
        {isOperatorVerified && (
          <Polyline
            positions={mitigated_prediction_track}
            pathOptions={{
              color: '#f59e0b',
              weight: 2,
              dashArray: '4 6',
              opacity: 0.45,
            }}
          >
            <Tooltip sticky>
              <span className="font-mono text-xs">
                Pre-Override Automated Trajectory (Superseded by Watch Officer)
              </span>
            </Tooltip>
          </Polyline>
        )}

        {/* Active Mitigated / Operator-Verified Safe Trajectory */}
        <Polyline
          positions={activeMitigatedTrack}
          pathOptions={{
            color: isOperatorVerified
              ? '#22d3ee'
              : isTugRareManeuver && mitigated
                ? '#f59e0b'
                : '#22d3ee',
            weight: isOperatorVerified ? 4.5 : mitigated ? 4 : 2,
            opacity: isOperatorVerified || mitigated ? 1.0 : 0.4,
          }}
        >
          <Tooltip sticky>
            <span className="font-mono text-xs">
              {isOperatorVerified
                ? 'OPERATOR VERIFIED Safe Trajectory (Watch Officer Override)'
                : isTugRareManeuver
                  ? 'Mitigated Trajectory (Lagging Sharp Turn due to False-Positive Gate Rejection)'
                  : 'Mitigated Anchor-Verified Trajectory (Tier-1 Audited)'}
            </span>
          </Tooltip>
        </Polyline>

        {/* Decision Origin Marker (t=0) with OPERATOR VERIFIED Badge */}
        <CircleMarker
          center={vesselCurrentPos}
          radius={isOperatorVerified ? 8.5 : 7}
          pathOptions={{
            color: '#ffffff',
            fillColor: isOperatorVerified
              ? '#10b981'
              : isTugRareManeuver
                ? '#f59e0b'
                : '#22d3ee',
            fillOpacity: 1,
            weight: 2.5,
          }}
        >
          <Tooltip permanent direction="top" offset={[0, -10]}>
            <div className="font-mono text-[11px] font-bold flex items-center gap-1.5">
              <span>
                {hitl_alert.vessel_name} ({hitl_alert.mmsi})
              </span>
              {isOperatorVerified ? (
                <span className="rounded bg-emerald-600 px-1.5 py-0.5 text-[10px] text-white uppercase tracking-wider">
                  OPERATOR VERIFIED
                </span>
              ) : isTugRareManeuver ? (
                <span className="rounded bg-amber-600 px-1.5 py-0.5 text-[10px] text-slate-950 uppercase tracking-wider">
                  HIGH-G TURN (64.2°/m)
                </span>
              ) : (
                <span className="text-slate-400 font-normal">(t=0)</span>
              )}
            </div>
          </Tooltip>
        </CircleMarker>

        {/* Terminal Horizon Markers (t+15m) */}
        <CircleMarker
          center={gtEndPos}
          radius={5}
          pathOptions={{ color: '#022c22', fillColor: '#34d399', fillOpacity: 1, weight: 1.5 }}
        >
          <Tooltip direction="left">
            <span className="font-mono text-xs">Actual Position (t+15m)</span>
          </Tooltip>
        </CircleMarker>

        <CircleMarker
          center={unmitEndPos}
          radius={5}
          pathOptions={{ color: '#4c0519', fillColor: '#f43f5e', fillOpacity: 0.9, weight: 1.5 }}
        >
          <Tooltip direction="right">
            <span className="font-mono text-xs">
              Unmitigated t+15m Forecast (Divergence: {scenario.hitl_alert.tier1_vs_tier2_divergence_m}m)
            </span>
          </Tooltip>
        </CircleMarker>

        {(mitigated || isOperatorVerified) && (
          <CircleMarker
            center={mitEndPos}
            radius={6}
            pathOptions={{
              color: isOperatorVerified ? '#064e3b' : '#083344',
              fillColor: isOperatorVerified
                ? '#10b981'
                : isTugRareManeuver
                  ? '#f59e0b'
                  : '#22d3ee',
              fillOpacity: 1,
              weight: 2,
            }}
          >
            <Tooltip direction="bottom">
              <span className="font-mono text-xs">
                {isOperatorVerified
                  ? 'OPERATOR VERIFIED Safe Endpoint (t+15m)'
                  : isTugRareManeuver
                    ? 'Trust-Gated Endpoint (Lagging Escort Turn)'
                    : 'Anchor-Verified t+15m Forecast'}
              </span>
            </Tooltip>
          </CircleMarker>
        )}
      </MapContainer>

      {/* Bottom-Left Tactical Map Legend Overlay */}
      <div className="absolute bottom-4 left-4 z-[500] w-84 rounded-lg border border-slate-800/95 bg-slate-900/95 p-3 shadow-xl backdrop-blur-md">
        <div className="flex items-center justify-between border-b border-slate-800 pb-1.5 mb-2">
          <span className="flex items-center gap-1.5 text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-200">
            <Layers className="h-3.5 w-3.5 text-cyan-400" />
            Tactical Track &amp; Feed Legend
          </span>
          <span className="flex items-center gap-1 text-[10px] font-mono text-slate-400">
            <Compass className="h-3 w-3 text-slate-400" />
            NORFOLK APPROACH
          </span>
        </div>

        <div className="space-y-1.5 text-xs">
          <div className="flex items-center gap-2.5">
            <span className="h-1 w-6 rounded bg-emerald-400 inline-block shrink-0" />
            <span className="text-slate-200 font-medium">Ground-Truth Vessel Track</span>
          </div>
          <div className="flex items-center gap-2.5">
            <span className="h-0.5 w-6 border-b-2 border-dashed border-rose-500 inline-block shrink-0" />
            <span className="text-slate-200 font-medium">
              Unmitigated Prediction Track
            </span>
          </div>
          <div className="flex items-center gap-2.5">
            <span
              className={`h-1 w-6 rounded inline-block shrink-0 ${
                isOperatorVerified
                  ? 'bg-cyan-400 shadow-[0_0_8px_#10b981]'
                  : isTugRareManeuver && mitigated
                    ? 'bg-amber-400 shadow-[0_0_8px_#f59e0b]'
                    : 'bg-cyan-400 shadow-[0_0_8px_#22d3ee]'
              }`}
            />
            <span className="text-slate-200 font-medium">
              {isOperatorVerified
                ? 'Operator-Verified Safe Path (HITL Locked)'
                : isTugRareManeuver && mitigated
                  ? 'Trust-Gated Track (Lagging Sharp Turn)'
                  : 'Anchor-Verified Mitigated + 95% Cone'}
            </span>
          </div>
        </div>

        {isOperatorVerified && (
          <div className="mt-2 rounded border border-emerald-500/40 bg-emerald-950/35 px-2.5 py-1.5 text-[11px] font-mono text-emerald-300 flex items-center gap-1.5">
            <ShieldCheck className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
            <span>Watch Officer override active — safe corridor locked.</span>
          </div>
        )}

        <div className="mt-2.5 pt-2 border-t border-slate-800/90 space-y-1 text-[11px] font-mono text-slate-400">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1 text-emerald-300">
              <Radio className="h-3 w-3" /> Tier-1 USCG Anchor:
            </span>
            <span>Slow Batch (30-60m delay), High Trust</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1 text-amber-300">
              <Radio className="h-3 w-3" /> Tier-2 Commercial:
            </span>
            <span>Real-time, High Vol (Stale/Jitter Risk)</span>
          </div>
        </div>
      </div>
    </div>
  )
}
