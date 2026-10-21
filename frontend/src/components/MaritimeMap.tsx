import { Compass, Layers, Navigation, Radio } from 'lucide-react'
import { CircleMarker, MapContainer, Polygon, Polyline, TileLayer, Tooltip } from 'react-leaflet'
import { Badge } from '@/components/ui/badge'
import type { ScenarioData } from '@/types/maritime'

interface MaritimeMapProps {
  scenario: ScenarioData
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

export function MaritimeMap({ scenario }: MaritimeMapProps) {
  const {
    ground_truth_track,
    unmitigated_prediction_track,
    mitigated_prediction_track,
    uncertainty_polygon,
    mitigated,
    condition,
    window,
    hitl_alert,
  } = scenario

  const vesselCurrentPos = ground_truth_track[2]
  const gtEndPos = ground_truth_track[ground_truth_track.length - 1]
  const unmitEndPos = unmitigated_prediction_track[unmitigated_prediction_track.length - 1]
  const mitEndPos = mitigated_prediction_track[mitigated_prediction_track.length - 1]

  return (
    <div className="relative h-full w-full bg-slate-950 overflow-hidden select-none">
      {/* Top-Left Tactical HUD Overlay */}
      <div className="absolute top-3 left-14 z-[500] flex flex-wrap items-center gap-2 pointer-events-none">
        <div className="rounded-md border border-slate-800/90 bg-slate-900/90 px-3 py-1.5 backdrop-blur-md shadow-lg">
          <div className="flex items-center gap-2 text-xs font-mono text-slate-200">
            <Navigation className="h-3.5 w-3.5 text-cyan-400" />
            <span className="font-bold">{hitl_alert.vessel_name}</span>
            <span className="text-slate-400">MMSI {hitl_alert.mmsi}</span>
          </div>
          <div className="mt-0.5 flex items-center gap-3 text-[11px] font-mono text-slate-400">
            <span>CORRIDOR: THIMBLE SHOAL INBOUND</span>
            <span>•</span>
            <span>TRUST SCORE: {(hitl_alert.trust_score * 100).toFixed(0)}%</span>
          </div>
        </div>

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
      </div>

      {/* Leaflet Map Canvas */}
      <MapContainer
        center={[36.985, -76.12]}
        zoom={11}
        className="h-full w-full z-0"
        zoomControl={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
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
          positions={uncertainty_polygon}
          pathOptions={{
            color: mitigated ? '#22d3ee' : '#f43f5e',
            fillColor: mitigated ? '#22d3ee' : '#f43f5e',
            fillOpacity: mitigated ? 0.16 : 0.12,
            weight: 1,
            dashArray: '3 3',
          }}
        >
          <Tooltip sticky>
            <div className="font-mono text-xs">
              95% Quantile Prediction Cone ({mitigated ? 'Anchor-Calibrated' : 'Unmitigated Spread'})
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
            <span className="font-mono text-xs">Ground-Truth Vessel Trajectory (USCG Reference)</span>
          </Tooltip>
        </Polyline>

        {/* Unmitigated Poisoned Prediction Track (Rose Dashed Line) */}
        <Polyline
          positions={unmitigated_prediction_track}
          pathOptions={{
            color: '#f43f5e',
            weight: mitigated ? 2.5 : 4,
            dashArray: '8 6',
            opacity: mitigated ? 0.65 : 0.95,
          }}
        >
          <Tooltip sticky>
            <span className="font-mono text-xs">
              Unmitigated Model Prediction (W{window} {condition})
            </span>
          </Tooltip>
        </Polyline>

        {/* Mitigated Anchor-Verified Prediction Track (Cyan Solid Line) */}
        <Polyline
          positions={mitigated_prediction_track}
          pathOptions={{
            color: '#22d3ee',
            weight: mitigated ? 4 : 2,
            opacity: mitigated ? 1.0 : 0.4,
          }}
        >
          <Tooltip sticky>
            <span className="font-mono text-xs">
              Mitigated Anchor-Verified Trajectory (Tier-1 Audited)
            </span>
          </Tooltip>
        </Polyline>

        {/* Decision Origin Marker (t=0) */}
        <CircleMarker
          center={vesselCurrentPos}
          radius={7}
          pathOptions={{
            color: '#ffffff',
            fillColor: '#22d3ee',
            fillOpacity: 1,
            weight: 2,
          }}
        >
          <Tooltip permanent direction="top" offset={[0, -8]}>
            <span className="font-mono text-[11px] font-bold">
              MV VIRGINIA TRADER (t=0 Decision Point)
            </span>
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
              Unmitigated t+15m Forecast (Drift: {scenario.hitl_alert.tier1_vs_tier2_divergence_m}m)
            </span>
          </Tooltip>
        </CircleMarker>

        {mitigated && (
          <CircleMarker
            center={mitEndPos}
            radius={5.5}
            pathOptions={{ color: '#083344', fillColor: '#22d3ee', fillOpacity: 1, weight: 2 }}
          >
            <Tooltip direction="bottom">
              <span className="font-mono text-xs">Anchor-Verified t+15m Forecast</span>
            </Tooltip>
          </CircleMarker>
        )}
      </MapContainer>

      {/* Bottom-Left Tactical Map Legend Overlay */}
      <div className="absolute bottom-4 left-4 z-[500] w-80 rounded-lg border border-slate-800/95 bg-slate-900/95 p-3 shadow-xl backdrop-blur-md">
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
              Unmitigated Poisoned Prediction
            </span>
          </div>
          <div className="flex items-center gap-2.5">
            <span className="h-1 w-6 rounded bg-cyan-400 inline-block shrink-0 shadow-[0_0_8px_#22d3ee]" />
            <span className="text-slate-200 font-medium">
              Anchor-Verified Mitigated + 95% Cone
            </span>
          </div>
        </div>

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
