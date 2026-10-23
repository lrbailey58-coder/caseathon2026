"""Check Person 1's database BEFORE training, so bad data is caught early.

Run from the repository root:
    uv run python -m backend.ml.check_data

Each check prints PASS, WARN (worth knowing, training can continue) or FAIL
(fix with Person 1 before training). The full report is saved to
results/data_check_real.json. Exit code is 1 if anything FAILs.

This script is an audit, so it may read `is_injected_defect`; training code never does.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from backend.ml.data import CONDITIONS, DB_PATH, MAX_PLAUSIBLE_SOG_KN, REPO_ROOT, connect

# Chesapeake / Norfolk box used by Person 1's ingestion (backend/database/build_database.py)
LAT_RANGE, LON_RANGE = (36.8, 37.3), (-76.6, -75.8)
EXPECTED_COLUMNS = {
    "mmsi", "voyage_id", "timestamp", "lat", "lon", "sog", "cog", "vessel_type",
    "window_id", "source_tier", "condition", "is_injected_defect",
}


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail) -> None:
        self.checks.append({"check": name, "status": status, "detail": detail})
        print(f"[{status:4}] {name}: {detail}")

    def check(self, name: str, ok: bool, detail, fail_status: str = "FAIL") -> None:
        self.add(name, "PASS" if ok else fail_status, detail)


def run_checks() -> Report:
    r = Report()
    with connect(DB_PATH) as con:
        one = lambda sql: con.execute(sql).fetchone()[0]

        # --- 1. Structure --------------------------------------------------------
        tables = {t for (t,) in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
        r.check("tables exist", {"raw_reference_ais", "injected_streams"} <= tables, sorted(tables))
        cols = {c for (c,) in con.execute(
            "SELECT column_name FROM duckdb_columns() WHERE table_name = 'injected_streams'").fetchall()}
        r.check("injected_streams has the agreed columns", EXPECTED_COLUMNS <= cols,
                {"missing": sorted(EXPECTED_COLUMNS - cols)})

        counts = dict(con.execute(
            "SELECT condition, COUNT(*) FROM injected_streams GROUP BY 1").fetchall())
        raw_rows = one("SELECT COUNT(*) FROM raw_reference_ais")
        r.check("all 4 conditions present, same row count as raw data",
                set(counts) == set(CONDITIONS) and set(counts.values()) == {raw_rows},
                {"raw_reference_ais": raw_rows, **counts})

        # --- 2. Values (clean data) ----------------------------------------------
        v = con.execute(f"""
            SELECT MIN(lat), MAX(lat), MIN(lon), MAX(lon), MIN(sog), MAX(sog), MIN(cog), MAX(cog),
                   SUM(CASE WHEN sog >= 102 OR cog >= 360 OR lat >= 91 OR lon >= 181 THEN 1 ELSE 0 END)
            FROM injected_streams WHERE condition = 'NOMINAL'""").fetchone()
        r.check("positions inside the Norfolk box",
                LAT_RANGE[0] <= v[0] and v[1] <= LAT_RANGE[1] and LON_RANGE[0] <= v[2] and v[3] <= LON_RANGE[1],
                {"lat": [v[0], v[1]], "lon": [v[2], v[3]]})
        too_fast = one(f"SELECT COUNT(*) FROM raw_reference_ais WHERE sog > {MAX_PLAUSIBLE_SOG_KN}")
        ok_ranges = v[4] >= 0 and v[6] >= 0 and v[7] < 360
        detail = {"sog_kn": [v[4], v[5]], "cog_deg": [v[6], v[7]],
                  f"pings_over_{MAX_PLAUSIBLE_SOG_KN:.0f}_kn (loader drops them)": too_fast}
        if ok_ranges and too_fast == 0:
            r.add("speed and heading in valid ranges", "PASS", detail)
        elif ok_ranges and too_fast <= 0.0001 * raw_rows:  # a few source glitches: log and drop
            r.add("speed and heading in valid ranges (a few source glitches)", "WARN", detail)
        else:
            r.add("speed and heading in valid ranges", "FAIL", detail)
        r.check("no AIS 'not available' codes (sog 102.3, cog 360, lat 91, lon 181)", v[8] == 0, int(v[8]))

        missing = con.execute("""
            SELECT SUM(CASE WHEN lat IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN lon IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN sog IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN cog IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN timestamp IS NULL OR voyage_id IS NULL THEN 1 ELSE 0 END)
            FROM injected_streams WHERE condition = 'NOMINAL'""").fetchone()
        missing = dict(zip(["lat", "lon", "sog", "cog", "timestamp_or_voyage"], map(int, missing)))
        r.check("missing values (loader drops these rows)", sum(missing.values()) < 0.01 * raw_rows,
                missing, fail_status="WARN" if sum(missing.values()) < 0.05 * raw_rows else "FAIL")

        # --- 3. Voyages and timing ------------------------------------------------
        dups = one("""SELECT COUNT(*) FROM (SELECT voyage_id, timestamp FROM injected_streams
                      WHERE condition = 'NOMINAL' GROUP BY ALL HAVING COUNT(*) > 1)""")
        r.check("duplicate timestamps within a voyage (features keep the first)", dups == 0, dups,
                fail_status="WARN")

        gaps = con.execute("""
            SELECT quantile_cont(g, 0.5), quantile_cont(g, 0.95), MAX(g) FROM (
              SELECT epoch(timestamp) - epoch(LAG(timestamp) OVER (PARTITION BY voyage_id ORDER BY timestamp)) g
              FROM injected_streams WHERE condition = 'NOMINAL') WHERE g IS NOT NULL""").fetchone()
        r.check("ping gaps within voyages <= 30 min (voyage rule)", gaps[2] <= 1800,
                {"median_s": gaps[0], "p95_s": gaps[1], "max_s": gaps[2]})

        dur = con.execute("""
            SELECT COUNT(*), SUM(CASE WHEN n < 2 THEN 1 ELSE 0 END), SUM(CASE WHEN mins >= 20 THEN 1 ELSE 0 END),
                   quantile_cont(mins, 0.5)
            FROM (SELECT voyage_id, COUNT(*) n, (epoch(MAX(timestamp)) - epoch(MIN(timestamp))) / 60 mins
                  FROM injected_streams WHERE condition = 'NOMINAL' GROUP BY 1)""").fetchone()
        r.check("enough voyages long enough for 15-min predictions (>= 20 min)", dur[2] >= 300,
                {"voyages": dur[0], "single_ping": int(dur[1]), "20_min_or_longer": int(dur[2]),
                 "median_minutes": round(dur[3], 1)})

        split = one("""SELECT COUNT(*) FROM (SELECT voyage_id FROM injected_streams WHERE condition = 'NOMINAL'
                       GROUP BY 1 HAVING COUNT(DISTINCT window_id) > 1)""")
        r.check("voyages with pings in two windows (loader uses first-ping window)", split == 0, split,
                fail_status="WARN")

        # --- 4. Windows -----------------------------------------------------------
        per_window = con.execute("""
            WITH v AS (SELECT voyage_id, arg_min(window_id, timestamp) w, MIN(timestamp) start, MAX(timestamp) stop,
                              ANY_VALUE(TRY_CAST(vessel_type AS INTEGER)) vt
                       FROM injected_streams WHERE condition = 'NOMINAL' GROUP BY 1)
            SELECT w, COUNT(*) voyages,
                   SUM(CASE WHEN vt BETWEEN 70 AND 79 THEN 1 ELSE 0 END) cargo,
                   SUM(CASE WHEN vt BETWEEN 80 AND 89 THEN 1 ELSE 0 END) tanker,
                   SUM(CASE WHEN vt IN (31, 32, 52) THEN 1 ELSE 0 END) tug,
                   MIN(start), MAX(start), MAX(stop)
            FROM v GROUP BY 1 ORDER BY 1""").fetchall()
        table = {f"W{w}": {"voyages": n, "cargo": c, "tanker": t, "tug": g,
                           "first_start": str(a), "last_start": str(b), "last_ping": str(e)}
                 for w, n, c, t, g, a, b, e in per_window}
        r.check("6 windows, each with >= 50 voyages", len(per_window) == 6 and min(x[1] for x in per_window) >= 50,
                {k: f"{x['voyages']} voyages ({x['cargo']} cargo, {x['tanker']} tanker, {x['tug']} tug)"
                 for k, x in table.items()})
        ordered = all(per_window[i][6] <= per_window[i + 1][5] for i in range(len(per_window) - 1))
        r.check("windows are in time order (by voyage start)", ordered,
                {k: [x["first_start"], x["last_start"]] for k, x in table.items()})
        overlap = any(per_window[i][7] > per_window[i + 1][5] for i in range(len(per_window) - 1))
        r.check("windows do not overlap in time (training code cuts overlaps anyway)", not overlap,
                "long voyages from one window keep going after the next window starts" if overlap else "ok",
                fail_status="WARN")
        cargo_tanker_share = sum(x[2] + x[3] for x in per_window) / sum(x[1] for x in per_window)
        r.check("cargo + tanker share of voyages", cargo_tanker_share >= 0.3,
                f"{cargo_tanker_share:.0%} cargo/tanker, rest tugs/towing", fail_status="WARN")

        # --- 5. Feeds and injected defects ----------------------------------------
        tiers = con.execute("""
            SELECT SUM(CASE WHEN n_pings >= 2 AND n_tiers = 2 THEN 1 ELSE 0 END), SUM(CASE WHEN n_pings >= 2 THEN 1 ELSE 0 END),
                   AVG(tier1_share)
            FROM (SELECT voyage_id, COUNT(*) n_pings, COUNT(DISTINCT source_tier) n_tiers,
                         AVG(CASE WHEN source_tier = 'TIER1_ANCHOR' THEN 1.0 ELSE 0.0 END) tier1_share
                  FROM injected_streams WHERE condition = 'NOMINAL' GROUP BY 1)""").fetchone()
        r.check("both feeds see every voyage with 2+ pings (needed by the trust gate)", tiers[0] == tiers[1],
                {"voyages_with_both": int(tiers[0]), "voyages_with_2plus_pings": int(tiers[1]),
                 "avg_tier1_share": round(tiers[2], 3)})

        nominal_same = one("""SELECT COUNT(*) FROM (
            SELECT voyage_id, timestamp, lat, lon, sog, cog FROM injected_streams WHERE condition = 'NOMINAL'
            EXCEPT ALL SELECT voyage_id, base_datetime, lat, lon, sog, cog FROM raw_reference_ais)""")
        r.check("NOMINAL equals the untouched reference data", nominal_same == 0, f"{nominal_same} differing rows")

        clean_t1 = one("SELECT COUNT(*) FROM injected_streams WHERE source_tier = 'TIER1_ANCHOR' AND is_injected_defect")
        r.check("TIER1_ANCHOR is never corrupted", clean_t1 == 0, f"{clean_t1} corrupted anchor rows")

        linkable = one("""SELECT COUNT(*) FROM injected_streams i
            LEFT JOIN (SELECT DISTINCT voyage_id, base_datetime FROM raw_reference_ais) r
              ON i.voyage_id = r.voyage_id AND i.timestamp = r.base_datetime
            WHERE r.voyage_id IS NULL""")
        r.check("every corrupted row links to its clean row (voyage_id + timestamp)", linkable == 0,
                f"{linkable} rows without a clean match")

        defects = con.execute("""
            SELECT condition, window_id, ROUND(100.0 * AVG(CAST(is_injected_defect AS INTEGER)), 1)
            FROM injected_streams WHERE source_tier = 'TIER2_FAST' GROUP BY ALL ORDER BY ALL""").fetchall()
        by_cond = {}
        for cond, w, pct in defects:
            by_cond.setdefault(cond, {})[f"W{w}"] = pct
        r.add("INFO", "PASS", "share of TIER2_FAST rows corrupted (%) by condition and window:")
        for cond in CONDITIONS:
            print(f"         {cond:12} {by_cond.get(cond)}")
        r.checks[-1]["detail"] = by_cond
        nominal_clean = all(p <= 0.5 for p in by_cond.get("NOMINAL", {}).values())
        r.check("NOMINAL has no corruption", nominal_clean, by_cond.get("NOMINAL"))
    return r


def main() -> None:
    print(f"Checking {DB_PATH}\n")
    report = run_checks()
    statuses = [c["status"] for c in report.checks]
    summary = {s: statuses.count(s) for s in ("PASS", "WARN", "FAIL")}
    out = REPO_ROOT / "results" / "data_check_real.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "database": DB_PATH.name,
        "checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summary,
        "checks": report.checks,
    }, indent=2, default=str))
    print(f"\nSummary: {summary}  ->  saved {out.relative_to(REPO_ROOT)}")
    sys.exit(1 if summary["FAIL"] else 0)


if __name__ == "__main__":
    main()
