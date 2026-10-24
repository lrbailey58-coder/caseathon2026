import type {
  ConditionType,
  HitlLogsData,
  LatLon,
  MatrixRunEntry,
  MatrixSummaryData,
  ScenarioData,
  VesselPresetType,
} from '@/types/maritime'

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

const TUG_RARE_MANEUVER_GROUND_TRUTH_TRACK: LatLon[] = [
  [36.952, -75.985],
  [36.961, -76.012],
  [36.9705, -76.041],
  [36.9788, -76.068],
  [36.9852, -76.091],
  [36.9775, -76.106],
  [36.9665, -76.118],
  [36.954, -76.1275],
  [36.942, -76.1355],
  [36.931, -76.142],
  [36.921, -76.1475],
  [36.912, -76.152],
]

const TUG_ANCHOR_LAG_TANGENT_TRACK: LatLon[] = [
  [36.952, -75.985],
  [36.961, -76.012],
  [36.9705, -76.041],
  [36.9788, -76.068],
  [36.9858, -76.0935],
  [36.9922, -76.119],
  [36.998, -76.1445],
  [37.0028, -76.1695],
  [37.0055, -76.194],
  [37.004, -76.218],
  [36.999, -76.241],
  [36.992, -76.263],
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
    unmitigated_mae: [604.4, 641.7, 653.6, 598.5, 664.6, 667.2],
    mitigated_mae: [604.4, 644.3, 653.9, 601.8, 669.5, 670.8],
    tail_95_unmitigated: [2109.1, 2305.3, 2331.0, 2183.1, 2421.6, 2454.3],
    tail_95_mitigated: [2109.1, 2306.0, 2340.9, 2207.6, 2463.7, 2487.0],
    corrupted_vol_pct: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    staleness_min: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    discarded_pct: [0.0, 1.91, 1.75, 1.82, 1.76, 1.86],
    deferred_pct: [0.0, 15.66, 17.37, 14.47, 18.18, 18.79],
  },
  STALE: {
    unmitigated_mae: [604.4, 641.7, 654.6, 609.6, 687.1, 690.0],
    mitigated_mae: [604.4, 644.3, 654.3, 603.3, 667.7, 669.3],
    tail_95_unmitigated: [2109.1, 2305.3, 2297.4, 2174.1, 2381.4, 2378.6],
    tail_95_mitigated: [2109.1, 2306.0, 2347.0, 2197.0, 2445.4, 2460.0],
    corrupted_vol_pct: [0.0, 0.0, 13.32, 26.49, 44.73, 61.15],
    staleness_min: [0.0, 1.83, 4.07, 6.96, 9.44, 9.45],
    discarded_pct: [0.0, 1.91, 6.53, 11.23, 16.55, 22.6],
    deferred_pct: [0.0, 15.66, 17.61, 14.78, 18.68, 18.82],
  },
  LOW_QUALITY: {
    unmitigated_mae: [604.4, 641.7, 661.8, 614.8, 678.2, 682.9],
    mitigated_mae: [604.4, 644.3, 651.2, 602.0, 666.4, 667.4],
    tail_95_unmitigated: [2109.1, 2305.3, 2360.9, 2252.0, 2458.8, 2468.0],
    tail_95_mitigated: [2109.1, 2306.0, 2335.7, 2201.5, 2442.7, 2444.1],
    corrupted_vol_pct: [0.0, 0.0, 100.0, 100.0, 100.0, 100.0],
    staleness_min: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    discarded_pct: [0.0, 1.91, 7.31, 9.35, 11.29, 16.41],
    deferred_pct: [0.0, 15.66, 17.53, 14.49, 18.83, 19.6],
  },
  COMBINED: {
    unmitigated_mae: [604.4, 641.7, 661.4, 619.8, 699.6, 707.4],
    mitigated_mae: [604.4, 644.3, 655.4, 603.1, 665.6, 665.2],
    tail_95_unmitigated: [2109.1, 2305.3, 2327.3, 2206.4, 2421.2, 2414.7],
    tail_95_mitigated: [2109.1, 2306.0, 2342.8, 2198.3, 2425.2, 2428.6],
    corrupted_vol_pct: [0.0, 0.0, 100.0, 100.0, 100.0, 100.0],
    staleness_min: [0.0, 1.83, 4.07, 6.98, 9.44, 9.45],
    discarded_pct: [0.0, 1.91, 11.36, 17.34, 23.43, 30.92],
    deferred_pct: [0.0, 15.66, 17.62, 14.91, 18.61, 18.93],
  },
}

const WINDOW_LABELS: Record<number, string> = {
  1: 'W1: Baseline Reference',
  2: 'W2: Early Feed Drift',
  3: 'W3: Moderate Poisoning',
  4: 'W4: Heavy Retraining Drift',
  5: 'W5: Severe Channel Excursion',
  6: 'W6: Unseen Test Period',
}

const CONDITION_LABELS: Record<ConditionType, string> = {
  NOMINAL: 'Reference / Nominal',
  STALE: 'Stale Only',
  LOW_QUALITY: 'Low-Quality Only',
  COMBINED: 'Combined Poisoning',
}

export function buildFallbackScenario(
  condition: ConditionType,
  mitigated: boolean,
  window: number,
  vesselType: VesselPresetType = 'CARGO_CONTAINER',
): ScenarioData {
  const wIdx = Math.max(0, Math.min(5, window - 1))
  const driftScale = condition === 'NOMINAL' ? 0 : wIdx / 5
  const table = ERROR_PROGRESSION[condition]

  const isTug = vesselType === 'HARBOR_TUG_RARE_MANEUVER'
  const gtTrack = isTug ? TUG_RARE_MANEUVER_GROUND_TRUTH_TRACK : BASE_GROUND_TRUTH_TRACK

  const unmitigatedTrack: LatLon[] = []
  const mitigatedTrack: LatLon[] = []
  const upperBound: LatLon[] = []
  const lowerBound: LatLon[] = []

  gtTrack.forEach(([lat, lon], i) => {
    const progress = Math.max(0, (i - 2) / (gtTrack.length - 3))
    if (i <= 2) {
      unmitigatedTrack.push([lat, lon])
      mitigatedTrack.push([lat, lon])
      return
    }

    if (isTug) {
      let dLat = 0
      let dLon = 0
      if (condition === 'NOMINAL') {
        dLat = -0.0008 * progress
        dLon = 0.0006 * progress
      } else if (condition === 'STALE') {
        dLat = 0.0095 * driftScale * progress ** 1.2
        dLon = 0.008 * driftScale * progress
      } else if (condition === 'LOW_QUALITY') {
        const osc = i % 2 === 0 ? 0.0028 : -0.0022
        dLat = osc * Math.max(0.35, driftScale)
        dLon = 0.0055 * driftScale * progress
      } else {
        const osc = i % 2 === 0 ? 0.0032 : -0.0024
        dLat = 0.011 * driftScale * progress + osc * Math.max(0.4, driftScale)
        dLon = 0.0095 * driftScale * progress
      }

      const unmitLat = Number((lat + dLat).toFixed(6))
      const unmitLon = Number((lon + dLon).toFixed(6))
      unmitigatedTrack.push([unmitLat, unmitLon])

      const [lagLat, lagLon] = TUG_ANCHOR_LAG_TANGENT_TRACK[i]
      const mitLat = Number((lagLat - 0.0012 * progress).toFixed(6))
      const mitLon = Number((lagLon + 0.0008 * progress).toFixed(6))
      mitigatedTrack.push([mitLat, mitLon])

      const centerLat = mitigated ? mitLat : unmitLat
      const centerLon = mitigated ? mitLon : unmitLon
      const coneHalfWidth = (0.0024 + 0.0028 * Math.max(0.4, driftScale)) * (0.5 + 0.9 * progress)

      upperBound.push([Number((centerLat + coneHalfWidth).toFixed(6)), centerLon])
      lowerBound.push([Number((centerLat - coneHalfWidth).toFixed(6)), centerLon])
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
    gtTrack[0],
    ...upperBound,
    ...[...lowerBound].reverse(),
  ]

  const stalenessMin = isTug ? 0.6 : table.staleness_min[wIdx]
  const corruptedPct = table.corrupted_vol_pct[wIdx]
  const divergenceM = isTug
    ? Number((385 + 12 * wIdx).toFixed(1))
    : Number((table.unmitigated_mae[wIdx] * 0.88).toFixed(1))
  const trustScore = isTug
    ? 0.29
    : Number(Math.max(0.12, 0.98 - (corruptedPct / 100) * 0.85).toFixed(2))

  let explanation = ''
  if (isTug) {
    explanation = `[W${window} // FAILURE MODE] TUG CAPE HENRY (MMSI 367004890) executes a legitimate high-G emergency escort turn in Norfolk channel (COG delta = 64.2°/min > 45°/min gate threshold). Kinematic + Tier-1 Anchor Trust Gate falsely flags 82.4% of valid high-frequency Tier-2 turn pings as 'Low-Quality / Sensor Spike', causing the mitigated trajectory to lag ${divergenceM.toFixed(0)}m along the pre-turn anchor vector until Watch Officer manual waypoint verification.`
  } else if (condition === 'NOMINAL') {
    explanation =
      `Tier-2 commercial stream aligns within 45m of delayed Tier-1 USCG coastal anchor (${table.discarded_pct[wIdx].toFixed(1)}% filtered). Kinematic velocity and heading deltas remain within nominal Norfolk channel bounds.`
  } else if (condition === 'STALE') {
    explanation = `Tier-2 Fast Feed repeats cached vessel coordinates (avg age ${stalenessMin.toFixed(1)} min) with fresh timestamps, deviating ${divergenceM.toFixed(0)}m from Tier-1 USCG Anchor trajectory; Trust Gate filters ${table.discarded_pct[wIdx].toFixed(1)}% of Tier-2 records.`
  } else if (condition === 'LOW_QUALITY') {
    explanation = `Tier-2 Fast Feed exhibits severe GPS multipath jitter and SOG/COG spikes across ${corruptedPct.toFixed(0)}% of retraining buffer (${table.discarded_pct[wIdx].toFixed(1)}% filtered by Trust Gate); deviates ${divergenceM.toFixed(0)}m from Tier-1 Anchor.`
  } else {
    explanation = `Combined poisoning: Tier-2 Fast Feed has ${stalenessMin.toFixed(1)} min staleness and ${corruptedPct.toFixed(0)}% corrupted records (${table.discarded_pct[wIdx].toFixed(1)}% filtered by Trust Gate), pulling unmitigated t+15 prediction ${divergenceM.toFixed(0)}m outside Thimble Shoal inbound channel.`
  }

  const displacementMae = isTug
    ? mitigated
      ? 1065.9
      : 996.5
    : mitigated
      ? table.mitigated_mae[wIdx]
      : table.unmitigated_mae[wIdx]

  const tail95Error = isTug
    ? mitigated
      ? 3030.0
      : 2784.2
    : mitigated
      ? table.tail_95_mitigated[wIdx]
      : table.tail_95_unmitigated[wIdx]

  return {
    condition,
    mitigated,
    window,
    window_label: WINDOW_LABELS[window] ?? `W${window}`,
    vessel_type: vesselType,
    ground_truth_track: gtTrack,
    unmitigated_prediction_track: unmitigatedTrack,
    mitigated_prediction_track: mitigatedTrack,
    uncertainty_polygon: uncertaintyPolygon,
    metrics: {
      displacement_mae_m: displacementMae,
      tail_95_error_m: tail95Error,
      data_discarded_pct: isTug ? (mitigated ? 46.4 : 0) : mitigated ? table.discarded_pct[wIdx] : 0,
      decisions_deferred_pct: isTug ? (mitigated ? 54.5 : 0) : mitigated ? table.deferred_pct[wIdx] : 0,
      model_compute_latency_ms: mitigated ? 13.7 : 6.3,
      avg_human_review_s: 8.4,
    },
    timeline_series: [0, 1, 2, 3, 4, 5].map((i) => ({
      window: `W${i + 1}`,
      unmitigated_mae: table.unmitigated_mae[i],
      mitigated_mae: table.mitigated_mae[i],
      tier2_corrupted_volume_pct: table.corrupted_vol_pct[i],
      avg_data_staleness_min: table.staleness_min[i],
    })),
    hitl_alert: {
      vessel_name: isTug ? 'KENDALL J HEBERT' : 'GRANDE GUINEA',
      mmsi: isTug ? '367567990' : '247293300',
      vessel_type: vesselType,
      tier1_vs_tier2_divergence_m: divergenceM,
      staleness_minutes: stalenessMin,
      trust_score: trustScore,
      requires_human_review: isTug || (condition !== 'NOMINAL' && window >= 2),
      explanation,
      cog_delta_deg_per_min: isTug ? 112.2 : 18.5,
      clean_false_alarm_pct: isTug ? 16.1 : 0.0,
      false_positive_gate_flag: isTug,
    },
  }
}

export function buildFallbackHitlLogs(): HitlLogsData {
  return {
    total_interventions: 1,
    avg_human_verification_s: 8.4,
    model_compute_latency_ms: 13.9,
    logs: [
      {
        id: 1,
        timestamp: '2026-10-22T12:00:00Z',
        mmsi: '367591240',
        vessel_name: 'MV VIRGINIA TRADER',
        condition: 'COMBINED',
        window_id: 5,
        action_taken: 'DEFER_TO_ANCHOR_KINEMATIC',
        review_duration_seconds: 8.4,
        operator_notes:
          'Initial Watch Officer calibration verification against USCG Tier-1 coastal anchor.',
        model_compute_latency_ms: 14.2,
      },
    ],
  }
}

export function buildFallbackMatrixSummary(window: number = 6): MatrixSummaryData {
  const wIdx = Math.max(0, Math.min(5, window - 1))
  const conditions: ConditionType[] = ['NOMINAL', 'STALE', 'LOW_QUALITY', 'COMBINED']
  const runs: MatrixRunEntry[] = []
  const cleanUnmitMae = ERROR_PROGRESSION.NOMINAL.unmitigated_mae[wIdx]

  let runId = 1
  for (const cond of conditions) {
    const table = ERROR_PROGRESSION[cond]
    const unmitMae = table.unmitigated_mae[wIdx]
    const damage = unmitMae - cleanUnmitMae

    for (const isMitigated of [false, true]) {
      const mae = isMitigated ? table.mitigated_mae[wIdx] : unmitMae
      const tail95 = isMitigated
        ? table.tail_95_mitigated[wIdx]
        : table.tail_95_unmitigated[wIdx]
      const discarded = isMitigated ? table.discarded_pct[wIdx] : 0
      const deferred = isMitigated ? table.deferred_pct[wIdx] : 0
      const recoveryGain =
        isMitigated && cond !== 'NOMINAL' && damage > 1e-9
          ? Number((((unmitMae - mae) / damage) * 100).toFixed(1))
          : isMitigated && unmitMae > 0
            ? Number((((unmitMae - mae) / unmitMae) * 100).toFixed(1))
            : 0
      const latencyMs = isMitigated ? 13.7 : 6.3

      runs.push({
        run_id: runId++,
        condition: cond,
        condition_label: CONDITION_LABELS[cond],
        mitigated: isMitigated,
        defense_label: isMitigated
          ? 'With Countermeasure (Anchor + HITL)'
          : 'Without Countermeasure (Naive)',
        window,
        window_label: WINDOW_LABELS[window] ?? `W${window}`,
        displacement_mae_m: mae,
        tail_95_error_m: tail95,
        data_discarded_pct: discarded,
        decisions_deferred_pct: deferred,
        recovery_gain_pct: recoveryGain,
        model_compute_latency_ms: latencyMs,
      })
    }
  }

  return {
    window,
    window_label: WINDOW_LABELS[window] ?? `W${window}`,
    avg_human_review_s: 8.4,
    runs,
  }
}

export function buildFallbackExportBundle(hitlLogs: HitlLogsData): Record<string, unknown> {
  const w6Matrix = buildFallbackMatrixSummary(6)
  return {
    schema_version: '1.0.0',
    exported_at_utc: new Date().toISOString(),
    project: 'NEPTUNE-GUARD // W&M AI Case-a-thon 2026 (Unintentional Data Poisoning)',
    evaluation_unit: 'Complete chronological vessel voyages (15-minute ahead prediction horizon)',
    primary_metrics: ['displacement_mae_m', 'tail_95_error_m'],
    eight_condition_metrics_table_w6: w6Matrix.runs,
    dataset_manifest_metadata: {
      dataset_name: 'MarineCadastre AIS 2025 (Chesapeake Bay & Norfolk Corridor)',
      source_url: 'https://noaaocm.blob.core.windows.net/ais/csv2/csv2025',
      date_range_utc: ['2025-09-01 00:00:00', '2025-10-12 23:59:54'],
      spatial_bounding_box: { lat_min: 36.8, lat_max: 37.3, lon_min: -76.6, lon_max: -75.8 },
      vessel_cohorts: ['Tug/Towing (31, 32, 52)', 'Cargo (70-79)', 'Tanker (80-89)'],
      chronological_windows: 6,
      total_reference_pings: 671747,
      unique_voyages: 9680,
      unique_vessels: 630,
      license: 'Public Domain (U.S. Government NOAA/BOEM work)',
    },
    modification_log_summary: {
      total_rows: 671747,
      settings: {
        random_seed: 42,
        tier_rule: 'every 5th ping of each voyage -> TIER1_ANCHOR; rest -> TIER2_FAST',
        defects_applied_to: 'TIER2_FAST only, windows W2-W6; TIER1_ANCHOR and W1 stay clean',
        stale_rates_by_window: { 1: 0.0, 2: 0.15, 3: 0.3, 4: 0.5, 5: 0.7, 6: 0.7 },
        noise_sigma_deg_by_window: { 1: 0.0, 2: 0.0005, 3: 0.001, 4: 0.002, 5: 0.0035, 6: 0.0035 },
      },
    },
    hitl_oversight_and_latency_stats: {
      total_interventions: hitlLogs.total_interventions,
      avg_human_verification_seconds: hitlLogs.avg_human_verification_s,
      avg_model_compute_latency_ms: hitlLogs.model_compute_latency_ms,
      latency_separation_note:
        'Model compute latency (milliseconds) is tracked and reported strictly separately from Watch Officer human verification/override duration (seconds).',
      recent_audit_trail: hitlLogs.logs,
    },
    failure_mode_boundary_analysis: {
      preset_id: 'HARBOR_TUG_RARE_MANEUVER',
      source: 'real AIS voyage (W6, clean data, so every gate flag is a false alarm)',
      vessel: 'JANE MCALLISTER (MMSI 368312320, voyage 368312320_196)',
      max_turn_deg_per_min: 104.7,
      clean_false_alarm_pct: 20.3,
      adverse_tradeoff:
        '[W6 (Unseen Test Period) // FAILURE MODE] JANE MCALLISTER turns at up to 105 deg/min. Even on clean data the trust gate wrongly flags 20% of its valid fast-feed reports, because it compares them to a straight line between trusted reports ~6 min apart.',
    },
  }
}
