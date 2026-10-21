import type { ConditionType, LatLon, ScenarioData } from '@/types/maritime'

const BASE_GROUND_TRUTH_TRACK: LatLon[] = [
  [36.952, -75.985],
  [36.961, -76.012],
  [36.9705, -76.041],
  [36.979, -76.0695],
  [36.9865, -76.098],
  [36.9935, -76.126],
  [36.9995, -76.154],
  [37.004, -76.181],
  [37.0065, -76.208],
  [37.0045, -76.235],
  [36.9985, -76.261],
  [36.989, -76.286],
]

const ERROR_PROGRESSION: Record<
  ConditionType,
  {
    unmitigated_mae: number[]
    mitigated_mae: number[]
    tail_95_unmitigated: number[]
    tail_95_mitigated: number[]
    corrupted_vol_pct: number[]
    staleness_min: number[]
    discarded_pct: number[]
    deferred_pct: number[]
  }
> = {
  NOMINAL: {
    unmitigated_mae: [42.0, 44.5, 43.8, 45.2, 46.1, 47.0],
    mitigated_mae: [41.5, 42.8, 42.1, 43.0, 43.6, 44.2],
    tail_95_unmitigated: [88.0, 92.0, 91.0, 94.5, 96.0, 98.0],
    tail_95_mitigated: [85.0, 88.0, 87.5, 89.0, 90.5, 91.5],
    corrupted_vol_pct: [2.0, 2.5, 2.2, 2.8, 3.0, 3.1],
    staleness_min: [1.2, 1.3, 1.2, 1.4, 1.4, 1.5],
    discarded_pct: [1.8, 2.1, 2.0, 2.4, 2.6, 2.8],
    deferred_pct: [2.0, 2.0, 2.5, 2.5, 3.0, 3.0],
  },
  STALE: {
    unmitigated_mae: [43.0, 118.0, 215.0, 342.0, 468.0, 540.0],
    mitigated_mae: [42.0, 56.0, 68.0, 79.0, 88.0, 94.0],
    tail_95_unmitigated: [90.0, 245.0, 430.0, 685.0, 910.0, 1060.0],
    tail_95_mitigated: [86.0, 115.0, 138.0, 162.0, 179.0, 192.0],
    corrupted_vol_pct: [4.0, 18.0, 34.0, 51.0, 66.0, 72.0],
    staleness_min: [1.5, 6.8, 12.4, 17.5, 22.8, 25.4],
    discarded_pct: [3.2, 16.5, 31.0, 47.5, 61.2, 67.8],
    deferred_pct: [2.5, 6.0, 10.5, 14.0, 18.5, 21.0],
  },
  LOW_QUALITY: {
    unmitigated_mae: [44.0, 134.0, 248.0, 389.0, 512.0, 595.0],
    mitigated_mae: [42.5, 59.0, 74.0, 86.0, 96.0, 105.0],
    tail_95_unmitigated: [92.0, 290.0, 520.0, 790.0, 1045.0, 1210.0],
    tail_95_mitigated: [88.0, 124.0, 152.0, 178.0, 198.0, 215.0],
    corrupted_vol_pct: [5.0, 20.0, 38.0, 55.0, 68.0, 75.0],
    staleness_min: [1.4, 2.1, 2.8, 3.4, 3.9, 4.2],
    discarded_pct: [4.1, 18.2, 35.4, 51.0, 63.8, 70.2],
    deferred_pct: [3.0, 7.5, 12.0, 16.5, 20.0, 23.5],
  },
  COMBINED: {
    unmitigated_mae: [45.0, 165.0, 310.0, 495.0, 670.0, 785.0],
    mitigated_mae: [43.0, 64.0, 82.0, 98.0, 112.0, 124.0],
    tail_95_unmitigated: [95.0, 355.0, 640.0, 980.0, 1320.0, 1540.0],
    tail_95_mitigated: [89.0, 134.0, 170.0, 204.0, 230.0, 252.0],
    corrupted_vol_pct: [6.0, 24.0, 44.0, 62.0, 76.0, 82.0],
    staleness_min: [1.6, 8.2, 14.8, 19.6, 24.5, 27.8],
    discarded_pct: [5.0, 21.8, 40.5, 57.4, 71.0, 76.5],
    deferred_pct: [3.5, 9.0, 15.0, 21.5, 26.0, 29.5],
  },
}

const WINDOW_LABELS: Record<number, string> = {
  1: 'W1: Baseline Reference',
  2: 'W2: Early Feed Drift',
  3: 'W3: Moderate Poisoning',
  4: 'W4: Heavy Retraining Drift',
  5: 'W5: Severe Channel Excursion',
  6: 'W6: Unseen Evaluation Period',
}

export function buildFallbackScenario(
  condition: ConditionType,
  mitigated: boolean,
  window: number,
): ScenarioData {
  const wIdx = Math.max(0, Math.min(5, window - 1))
  const driftScale = condition === 'NOMINAL' ? 0 : wIdx / 5
  const table = ERROR_PROGRESSION[condition]

  const unmitigatedTrack: LatLon[] = []
  const mitigatedTrack: LatLon[] = []
  const upperBound: LatLon[] = []
  const lowerBound: LatLon[] = []

  BASE_GROUND_TRUTH_TRACK.forEach(([lat, lon], i) => {
    const progress = Math.max(0, (i - 2) / (BASE_GROUND_TRUTH_TRACK.length - 3))
    if (i <= 2) {
      unmitigatedTrack.push([lat, lon])
      mitigatedTrack.push([lat, lon])
      return
    }

    let dLat = 0
    let dLon = 0
    if (condition === 'NOMINAL') {
      dLat = 0.0005 * progress
      dLon = 0.0004 * progress
    } else if (condition === 'STALE') {
      dLat = 0.0145 * driftScale * progress ** 1.35
      dLon = 0.018 * driftScale * progress ** 1.25
    } else if (condition === 'LOW_QUALITY') {
      const osc = i % 2 === 0 ? 0.0035 : -0.0022
      dLat = (-0.0155 * progress ** 1.3 + osc) * driftScale
      dLon = 0.011 * driftScale * progress
    } else {
      const osc = i % 2 === 0 ? 0.003 : -0.0015
      dLat = (0.0195 * progress ** 1.35 + osc) * driftScale
      dLon = 0.0225 * driftScale * progress ** 1.25
    }

    const unmitLat = Number((lat + dLat).toFixed(6))
    const unmitLon = Number((lon + dLon).toFixed(6))
    unmitigatedTrack.push([unmitLat, unmitLon])

    const mitLat = Number((lat + dLat * 0.14).toFixed(6))
    const mitLon = Number((lon + dLon * 0.14).toFixed(6))
    mitigatedTrack.push([mitLat, mitLon])

    const centerLat = mitigated ? mitLat : unmitLat
    const centerLon = mitigated ? mitLon : unmitLon
    let coneHalfWidth = (0.0014 + 0.0022 * driftScale) * (0.45 + 0.85 * progress)
    if (!mitigated && condition !== 'NOMINAL') {
      coneHalfWidth *= 1.65
    }

    upperBound.push([Number((centerLat + coneHalfWidth).toFixed(6)), centerLon])
    lowerBound.push([Number((centerLat - coneHalfWidth).toFixed(6)), centerLon])
  })

  const uncertaintyPolygon: LatLon[] = [
    BASE_GROUND_TRUTH_TRACK[2],
    ...upperBound,
    ...[...lowerBound].reverse(),
  ]

  const stalenessMin = table.staleness_min[wIdx]
  const corruptedPct = table.corrupted_vol_pct[wIdx]
  const divergenceM = Number((table.unmitigated_mae[wIdx] * 0.88).toFixed(1))
  const trustScore = Number(Math.max(0.12, 0.98 - (corruptedPct / 100) * 0.85).toFixed(2))

  let explanation = ''
  if (condition === 'NOMINAL') {
    explanation =
      'Tier-2 commercial stream aligns within 45m of delayed Tier-1 USCG coastal anchor. Kinematic velocity and heading deltas remain within nominal Norfolk channel bounds.'
  } else if (condition === 'STALE') {
    explanation = `Tier-2 Fast Feed repeats cached vessel coordinates (avg age ${stalenessMin.toFixed(1)} min) with fresh timestamps, deviating ${divergenceM.toFixed(0)}m from Tier-1 USCG Anchor trajectory.`
  } else if (condition === 'LOW_QUALITY') {
    explanation = `Tier-2 Fast Feed exhibits severe GPS multipath jitter and SOG/COG spikes across ${corruptedPct.toFixed(0)}% of retraining buffer; deviates ${divergenceM.toFixed(0)}m from Tier-1 Anchor.`
  } else {
    explanation = `Combined poisoning: Tier-2 Fast Feed has ${stalenessMin.toFixed(1)} min staleness and ${corruptedPct.toFixed(0)}% corrupted records, pulling unmitigated t+15 prediction ${divergenceM.toFixed(0)}m outside Thimble Shoal inbound channel.`
  }

  return {
    condition,
    mitigated,
    window,
    window_label: WINDOW_LABELS[window] ?? `W${window}`,
    ground_truth_track: BASE_GROUND_TRUTH_TRACK,
    unmitigated_prediction_track: unmitigatedTrack,
    mitigated_prediction_track: mitigatedTrack,
    uncertainty_polygon: uncertaintyPolygon,
    metrics: {
      displacement_mae_m: mitigated ? table.mitigated_mae[wIdx] : table.unmitigated_mae[wIdx],
      tail_95_error_m: mitigated ? table.tail_95_mitigated[wIdx] : table.tail_95_unmitigated[wIdx],
      data_discarded_pct: mitigated ? table.discarded_pct[wIdx] : 0,
      decisions_deferred_pct: mitigated ? table.deferred_pct[wIdx] : 0,
      model_compute_latency_ms: Number((11.8 + (mitigated ? 1.6 : 0) + 0.2 * wIdx).toFixed(1)),
      avg_human_review_s: mitigated ? 8.4 : 0,
    },
    timeline_series: [0, 1, 2, 3, 4, 5].map((i) => ({
      window: `W${i + 1}`,
      unmitigated_mae: table.unmitigated_mae[i],
      mitigated_mae: table.mitigated_mae[i],
      tier2_corrupted_volume_pct: table.corrupted_vol_pct[i],
      avg_data_staleness_min: table.staleness_min[i],
    })),
    hitl_alert: {
      vessel_name: 'MV VIRGINIA TRADER',
      mmsi: '367591240',
      tier1_vs_tier2_divergence_m: divergenceM,
      staleness_minutes: stalenessMin,
      trust_score: trustScore,
      requires_human_review: condition !== 'NOMINAL' && window >= 2,
      explanation,
    },
  }
}
