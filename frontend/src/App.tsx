import { useMemo, useState, useEffect } from "react";
import "./App.css";
import MapView from "./components/MapView";
import tsoZones from "./data/tso_zones.json";

// Base URL for the backend service (hosted)
const API_BASE = "https://dspunituebingen-windproject.hf.space";
// const LOCAL_API_BASE = "http://127.0.0.1:5137"; // handy for local dev / debugging

/**
 * Convert a Date into a simple YYYY-MM-DD string (used by <input type="date">).
 */
function toISODate(d: Date) {
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0"); // JS months are 0-based
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

/**
 * Add `years` to a date without mutating the original instance.
 */
function addYears(date: Date, years: number) {
  const d = new Date(date);
  d.setFullYear(d.getFullYear() + years);
  return d;
}

/**
 * Lightweight email validation for UI gating.
 * (Not meant to replace backend validation.)
 */
function isValidEmail(email: string) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
}

/**
 * Convert a year_start timestamp into a numeric year (UTC to avoid TZ surprises).
 */
function yearFromYearStart(yearStart?: string | null): number | null {
  if (!yearStart) return null;
  const d = new Date(yearStart);
  const y = d.getUTCFullYear();
  return Number.isFinite(y) ? y : null;
}

/* ============================================================
   ✅ No-dependency point-in-polygon for GeoJSON Polygon/MultiPolygon
   Input is a GeoJSON FeatureCollection (Polygon or MultiPolygon).
   GeoJSON coordinates are [lng, lat].
   ============================================================ */

type Position = [number, number]; // [lng, lat]

/**
 * Ray-casting point-in-ring test.
 * Returns true if `point` is inside the ring.
 */
function pointInRing(point: Position, ring: Position[]): boolean {
  const [x, y] = point;
  let inside = false;

  // Walk edges (j -> i) around the ring
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];

    // Standard ray-casting intersection check
    const intersect =
      (yi > y) !== (yj > y) &&
      x < ((xj - xi) * (y - yi)) / (yj - yi + 0.0) + xi;

    if (intersect) inside = !inside;
  }

  return inside;
}

/**
 * GeoJSON Polygon:
 * - polygonCoords[0] is the outer ring
 * - polygonCoords[1..] are holes (inner rings)
 */
function pointInPolygon(point: Position, polygonCoords: Position[][]): boolean {
  if (!polygonCoords.length) return false;

  // Must be inside the outer ring...
  if (!pointInRing(point, polygonCoords[0])) return false;

  // ...and not inside any hole.
  for (let k = 1; k < polygonCoords.length; k++) {
    if (pointInRing(point, polygonCoords[k])) return false;
  }

  return true;
}

/**
 * GeoJSON MultiPolygon: a list of polygons; hit if inside any polygon.
 */
function pointInMultiPolygon(point: Position, multiCoords: Position[][][]): boolean {
  for (const poly of multiCoords) {
    if (pointInPolygon(point, poly)) return true;
  }
  return false;
}

// --- Polygon area helpers (planar "shoelace" area in lon/lat degrees)
// Only used to compare relative sizes when overlaps exist (not real km²).

/**
 * Absolute area of a ring via shoelace formula.
 * Works in "degree space" (good enough for tie-breaking).
 */
function ringAreaAbs(ring: Position[]): number {
  if (ring.length < 3) return 0;
  let sum = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [x1, y1] = ring[j];
    const [x2, y2] = ring[i];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}

/**
 * Polygon area = outer ring area - sum(holes).
 * Clamped to 0 to avoid negative due to malformed inputs.
 */
function polygonAreaAbs(polygonCoords: Position[][]): number {
  if (!polygonCoords.length) return 0;
  let area = ringAreaAbs(polygonCoords[0]);
  for (let k = 1; k < polygonCoords.length; k++) {
    area -= ringAreaAbs(polygonCoords[k]);
  }
  return Math.max(0, area);
}

/**
 * MultiPolygon area = sum of polygon areas.
 */
function multiPolygonAreaAbs(multiCoords: Position[][][]): number {
  let area = 0;
  for (const poly of multiCoords) {
    area += polygonAreaAbs(poly);
  }
  return area;
}

// ✅ Map TSO display name -> backend tso_id (0..3)
// Keep this in sync with the backend's enum/order.
const TSO_NAME_TO_ID: Record<string, number> = {
  "50Hertz": 0,
  "amprion": 1,
  "TenneT": 2,
  "TransnetBW": 3,
};

/**
 * Determine the TSO zone for a given lat/lng.
 * If multiple polygons match (overlaps), pick the smallest area as "most specific".
 */
function getTsoIdFromPoint(lat: number, lng: number): number | null {
  const zones: any = tsoZones as any;
  if (!zones?.features?.length) return null;

  const pt: Position = [lng, lat]; // GeoJSON uses [lng, lat]

  // Collect all matching features, then resolve overlap via area tie-break.
  const matches: { tsoName: string; tsoId: number; area: number }[] = [];

  for (const f of zones.features) {
    const geom = f?.geometry;
    if (!geom) continue;

    const tsoName = String(f?.properties?.tso ?? "").trim();
    const tsoId = TSO_NAME_TO_ID[tsoName];
    if (!Number.isFinite(tsoId)) continue;

    let hit = false;
    let area = 0;

    if (geom.type === "Polygon") {
      const coords = geom.coordinates as Position[][];
      hit = pointInPolygon(pt, coords);
      if (hit) area = polygonAreaAbs(coords);
    } else if (geom.type === "MultiPolygon") {
      const coords = geom.coordinates as Position[][][];
      hit = pointInMultiPolygon(pt, coords);
      if (hit) area = multiPolygonAreaAbs(coords);
    }

    if (hit) {
      matches.push({ tsoName, tsoId, area });
    }
  }

  if (matches.length === 0) return null;

  // Prefer smallest area => likely the most specific boundary in overlaps.
  matches.sort((a, b) => a.area - b.area);

  // (Optional debug)
  // console.log("TSO matches", { lat, lng, matches: matches.slice(0, 5) });

  return matches[0].tsoId;
}

type ResultRow = {
  year: number;
  year_start?: string | null;

  opex: number | null;
  debt_service: number | null;

  revenue_p10: number | null;
  revenue_p50: number | null;
  revenue_p90: number | null;

  profit_p10: number | null;
  profit_p50: number | null;
  profit_p90: number | null;
};

/**
 * Normalize backend responses into a stable table shape for the UI.
 * Supports:
 *  - new format: already flat with revenue/profit percentiles per row
 *  - old format: cross-joined percentiles (level_1 / level_1_profit)
 */
function normalizeYearlyTable(raw: any): ResultRow[] {
  if (!Array.isArray(raw)) return [];

  // New backend format: rows contain flat percentile keys directly.
  const looksNewFlat =
    raw.length > 0 &&
    ("revenue_p50" in raw[0] || "profit_p50" in raw[0]) &&
    ("revenue_p10" in raw[0] ||
      "profit_p10" in raw[0] ||
      "revenue_p90" in raw[0] ||
      "profit_p90" in raw[0]);

  if (looksNewFlat) {
    return raw
      .map((x: any) => ({
        year: Number(x.year),
        year_start: x.year_start ?? null,
        opex: x.opex ?? null,
        debt_service: x.debt_service ?? null,
        revenue_p10: x.revenue_p10 ?? null,
        revenue_p50: x.revenue_p50 ?? null,
        revenue_p90: x.revenue_p90 ?? null,
        profit_p10: x.profit_p10 ?? null,
        profit_p50: x.profit_p50 ?? null,
        profit_p90: x.profit_p90 ?? null,
      }))
      .filter((r: ResultRow) => Number.isFinite(r.year))
      .sort((a: ResultRow, b: ResultRow) => a.year - b.year);
  }

  // Old backend format: multiple rows per year, percentiles stored in separate keys.
  const byYear = new Map<number, any[]>();
  for (const row of raw) {
    const y = Number(row.year ?? row.project_year);
    if (!Number.isFinite(y)) continue;
    if (!byYear.has(y)) byYear.set(y, []);
    byYear.get(y)!.push(row);
  }

  // Helper: choose the first finite numeric value (or null if none).
  const pickFirstFinite = (vals: any[]) => {
    for (const v of vals) {
      const n = v === null || v === undefined ? null : Number(v);
      if (n !== null && Number.isFinite(n)) return n;
    }
    return null;
  };

  const out: ResultRow[] = [];

  for (const [year, rows] of Array.from(byYear.entries()).sort((a, b) => a[0] - b[0])) {
    // Prefer an explicit year_start if present (some backends compute using dates)
    const year_start = rows.find((r) => r.year_start)?.year_start ?? null;

    // OPEX / debt are typically constant per year (but may appear repeated)
    const opex = pickFirstFinite(rows.map((r) => r.opex));
    const debt_service = pickFirstFinite(rows.map((r) => r.debt_service));

    // Percentile rows are identified via "level_1" and "level_1_profit"
    const revenueRows = rows.filter((r) => r.level_1 === "p10" || r.level_1 === "p50" || r.level_1 === "p90");
    const profitRows = rows.filter(
      (r) => r.level_1_profit === "p10" || r.level_1_profit === "p50" || r.level_1_profit === "p90"
    );

    // Revenue percentiles
    const revenue_p10 = pickFirstFinite(revenueRows.filter((r) => r.level_1 === "p10").map((r) => r.revenue_eur));
    const revenue_p50 = pickFirstFinite(revenueRows.filter((r) => r.level_1 === "p50").map((r) => r.revenue_eur));
    const revenue_p90 = pickFirstFinite(revenueRows.filter((r) => r.level_1 === "p90").map((r) => r.revenue_eur));

    // Profit percentiles (after OPEX + debt)
    const profit_p10 = pickFirstFinite(
      profitRows.filter((r) => r.level_1_profit === "p10").map((r) => r.profit_after_opex_and_debt_eur)
    );
    const profit_p50 = pickFirstFinite(
      profitRows.filter((r) => r.level_1_profit === "p50").map((r) => r.profit_after_opex_and_debt_eur)
    );
    const profit_p90 = pickFirstFinite(
      profitRows.filter((r) => r.level_1_profit === "p90").map((r) => r.profit_after_opex_and_debt_eur)
    );

    out.push({
      year,
      year_start,
      opex,
      debt_service,
      revenue_p10,
      revenue_p50,
      revenue_p90,
      profit_p10,
      profit_p50,
      profit_p90,
    });
  }

  return out;
}

/** ---------- helpers for the modal "top statistics" ---------- **/

type SummaryRow = {
  metric?: string;
  mean?: number | null;
  p10?: number | null;
  p50?: number | null;
  p90?: number | null;
};

/**
 * Pull a metric mean from final_summary_table (human-readable metric names).
 * Returns null if not found or not numeric.
 */
function pickMetricMeanFromFinalSummary(finalSummary: any, metricName: string): number | null {
  if (!Array.isArray(finalSummary)) return null;
  const row = (finalSummary as SummaryRow[]).find((r) => String(r.metric) === metricName);
  const v = row?.mean;
  const n = v === null || v === undefined ? null : Number(v);
  return n !== null && Number.isFinite(n) ? n : null;
}

/**
 * Pull a metric mean from stats_table (machine-readable metric keys).
 * Returns null if not found or not numeric.
 */
function pickMetricMeanFromStats(statsTable: any, metricKey: string): number | null {
  if (!Array.isArray(statsTable)) return null;
  const row = statsTable.find((r: any) => String(r.metric) === metricKey);
  const v = row?.mean;
  const n = v === null || v === undefined ? null : Number(v);
  return n !== null && Number.isFinite(n) ? n : null;
}

/**
 * Compact currency formatting for KPI cards.
 * Uses rough suffixes to keep the UI tight.
 */
function fmtEURCompact(v: number | null) {
  if (v === null) return "—";
  const abs = Math.abs(v);
  if (abs >= 1e9) return `€${(v / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `€${(v / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `€${(v / 1e3).toFixed(1)}k`;
  return `€${v.toFixed(0)}`;
}

/**
 * Format a decimal as a percent string.
 * If alreadyPercent is true, `v` is treated as 12.3 for 12.3%.
 */
function fmtPercent(v: number | null, alreadyPercent: boolean) {
  if (v === null) return "—";
  const p = alreadyPercent ? v : v * 100;
  return `${p.toFixed(2)}%`;
}

/** ---------- chart + table ---------- **/

/**
 * Small SVG line chart for quick trend inspection.
 * Draws three series and uses a shared scale based on combined extents.
 */
function MiniLineChart({
  title,
  seriesA,
  seriesB,
  seriesC,
  labelA,
  labelB,
  labelC,
}: {
  title: string;
  seriesA: { x: number; y: number }[];
  seriesB: { x: number; y: number }[];
  seriesC: { x: number; y: number }[];
  labelA: string;
  labelB: string;
  labelC: string;
}) {
  // Fixed viewBox keeps rendering predictable across layouts
  const width = 900;
  const height = 260;

  // Internal padding gives space for axes labels
  const padL = 56;
  const padR = 18;
  const padT = 18;
  const padB = 34;

  // Combine all points to compute global min/max for scales
  const all = [...seriesA, ...seriesB, ...seriesC].filter((p) => Number.isFinite(p.y));
  if (all.length === 0) {
    return (
      <div className="border rounded p-3 bg-white">
        <div className="text-sm font-medium mb-2">{title}</div>
        <div className="text-xs text-gray-600">No chart data available.</div>
      </div>
    );
  }

  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);

  // Prevent divide-by-zero for degenerate cases (e.g., all years identical)
  const xSpan = Math.max(1, xMax - xMin);
  const ySpan = Math.max(1, yMax - yMin);

  // Scale functions: data -> SVG pixels
  const xToPx = (x: number) => padL + ((x - xMin) / xSpan) * (width - padL - padR);
  const yToPx = (y: number) => padT + (1 - (y - yMin) / ySpan) * (height - padT - padB);

  /**
   * Convert points into an SVG path string.
   * Ensures stable ordering (by x) and drops non-finite values.
   */
  const toPath = (pts: { x: number; y: number }[]) => {
    const clean = pts
      .filter((p) => Number.isFinite(p.y))
      .sort((a, b) => a.x - b.x);
    if (clean.length === 0) return "";
    return clean.map((p, i) => `${i === 0 ? "M" : "L"} ${xToPx(p.x)} ${yToPx(p.y)}`).join(" ");
  };

  const pathA = toPath(seriesA);
  const pathB = toPath(seriesB);
  const pathC = toPath(seriesC);

  // Compact tick label formatting for readability in tight spaces
  const fmt = (v: number) => {
    const abs = Math.abs(v);
    if (abs >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
    if (abs >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
    return v.toFixed(0);
  };

  // Simple evenly-spaced y ticks (no fancy rounding; keeps code small)
  const yTicks = 4;
  const yTickVals = Array.from({ length: yTicks + 1 }, (_, i) => yMin + (i * ySpan) / yTicks);

  const xLeft = xMin;
  const xRight = xMax;

  return (
    <div className="border rounded p-3 bg-white">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-medium">{title}</div>

        {/* Legend: uses colored mini-lines to match paths */}
        <div className="flex items-center gap-3 text-xs text-gray-700">
          <span className="inline-flex items-center gap-1">
            <span className="inline-block w-3 h-[2px] bg-blue-600" />
            {labelA}
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="inline-block w-3 h-[2px] bg-red-600" />
            {labelB}
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="inline-block w-3 h-[2px] bg-emerald-600" />
            {labelC}
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
        <rect x={0} y={0} width={width} height={height} fill="white" />

        {/* Horizontal grid lines + y tick labels */}
        {yTickVals.map((yv, i) => {
          const y = yToPx(yv);
          return (
            <g key={i}>
              <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="#e5e7eb" />
              <text x={padL - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#6b7280">
                {fmt(yv)}
              </text>
            </g>
          );
        })}

        {/* X-axis endpoints */}
        <text x={xToPx(xLeft)} y={height - 10} textAnchor="start" fontSize="10" fill="#6b7280">
          {xLeft}
        </text>
        <text x={xToPx(xRight)} y={height - 10} textAnchor="end" fontSize="10" fill="#6b7280">
          {xRight}
        </text>

        {/* Series paths */}
        {pathA && <path d={pathA} fill="none" stroke="#2563eb" strokeWidth="2" />}
        {pathB && <path d={pathB} fill="none" stroke="#dc2626" strokeWidth="2" />}
        {pathC && <path d={pathC} fill="none" stroke="#059669" strokeWidth="2" />}
      </svg>

      {/* Small explanatory footer so users understand the derived values */}
      <div className="mt-2 text-xs text-gray-600">
        Revenue is estimated as <span className="font-mono">MWh × Market €/MWh</span>. Costs are{" "}
        <span className="font-mono">OPEX + Debt</span>.
      </div>
    </div>
  );
}

/**
 * Render a numeric cell value from a given ResultRow metric key.
 * Returns an em dash when missing/invalid (prevents "NaN" from leaking into UI).
 */
const formatCell = (metricKey: string, r: ResultRow) => {
  const v = (r as any)[metricKey] as number | null;
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
};

// The table is intentionally kept small: a few headline metrics per year.
const metricRows: { key: keyof ResultRow; label: string }[] = [
  { key: "revenue_p50", label: "Revenue (P50, €)" },
  { key: "opex", label: "OPEX (€)" },
  { key: "debt_service", label: "Debt service (€)" },
  { key: "profit_p50", label: "Profit (P50, €)" },
];

/**
 * Simple "static page" shown inside the app shell.
 * Keeps remarks/resources accessible without leaving the application.
 */
function RemarksResourcesPage({ onBack }: { onBack: () => void }) {
  // Curated list of sources and credits used for the tool and datasets.
  const resources: { title: string; url: string; note?: string }[] = [
    {
      title: "OpenStreetMap contributors",
      url: "https://www.openstreetmap.org/about",
    },
    {
      title: "Leaflet / React-Leaflet",
      url: "https://leafletjs.com/",
    },
    {
      title: "ERA5 hourly wind data",
      url: "https://www.kaggle.com/datasets/morteneghj/3-x-vestas-v100-2mw-pitch-power-windspeed",
    },
    {
      title: "Germany boundary",
      url: "https://www.naturalearthdata.com/",
    },
    {
      title: "Market electricity prices and wind production/consumption (national & per TSO)",
      url: "https://www.smard.de/en/downloadcenter/download-market-data/",
    },
    {
      title: "Tenders Bundesnetzagentur",
      url: "https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/Ausschreibungen/Wind_Onshore/BeendeteAusschreibungen/start.html",
    },
    {
      title: "Conceptual ideas",
      url: "https://www.sciencedirect.com/science/article/pii/S0960148115003183",
    },
    {
      title: "Turbine energy output",
      url: "https://www.kaggle.com/datasets/morteneghj/3-x-vestas-v100-2mw-pitch-power-windspeed",
    },
    {
      title: "TSO zone boundaries",
      url: "https://zenodo.org/records/7530196",
    },
  ];

  return (
    <div className="fixed inset-0 w-screen h-screen overflow-y-auto overflow-x-hidden bg-gray-50">
      <div className="max-w-3xl mx-auto px-6 py-10">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-semibold">Remarks & Resources</h1>
          <button type="button" onClick={onBack} className="px-3 py-2 rounded border bg-white hover:bg-gray-50 text-sm">
            ← Back to map
          </button>
        </div>

        <div className="bg-white rounded-xl border shadow-sm p-5 space-y-3">
          <h2 className="text-lg font-semibold">Remarks</h2>
          <p>
            Our results are indicative estimates rather than precise forecasts. They are based on limited publicly available
            datasets, historical averages, and simplified models, as well as approximations derived from selected academic
            publications and secondary data sources. Due to data constraints, several site-specific effects cannot be modeled
            in full detail and are represented only in an aggregated or approximate manner.
          </p>

          <p>
            Actual project performance can deviate materially from the results shown here. Factors such as micro-siting
            decisions, wake interactions, local terrain and roughness, wind variability, curtailment, grid connection
            constraints, permitting conditions, operational availability, maintenance strategies, and financing structures
            can all significantly influence energy production and financial outcomes.
          </p>

          <p>
            This tool is intended as a high-level screening and exploratory analysis to provide a broad overview of potential
            project characteristics. It should not be used as a standalone basis for investment decisions, financing
            applications, or commitments to stakeholders. Any project considered for development or financing should be
            validated through site-specific measurements, detailed engineering studies, and professional technical and
            financial due diligence.
          </p>
        </div>

        <div className="mt-6 bg-white rounded-xl border shadow-sm p-5">
          <h2 className="text-lg font-semibold mb-3">Resources</h2>
          <ul className="list-disc pl-5 space-y-2 text-sm text-gray-700">
            {resources.map((r, idx) => (
              <li key={idx}>
                <a href={r.url} target="_blank" rel="noopener noreferrer" className="font-medium text-blue-600 hover:underline">
                  {r.title}
                </a>
                {r.note ? <span className="text-gray-600"> — {r.note}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function App() {
  // Map pin state (defaults to roughly Germany center)
  const [latitude, setLatitude] = useState<number>(51.1657);
  const [longitude, setLongitude] = useState<number>(10.4515);

  // Main project inputs
  const [turbines, setTurbines] = useState<number>(2);
  const [turbineType, setTurbineType] = useState<string>("Nordex N149/5.X");
  const [hubHeight, setHubHeight] = useState<number>(160);
  const [emailSentModalOpen, setEmailSentModalOpen] = useState<boolean>(false);

  // Will be overwritten by auto-detection when pin moves
  const [tsoId, setTsoId] = useState<number>(1);

  // Auto-detect tsoId from pin (lat/lng) using GeoJSON zones
  useEffect(() => {
    const found = getTsoIdFromPoint(latitude, longitude);
    if (found !== null && found !== tsoId) setTsoId(found);
  }, [latitude, longitude, tsoId]);

  const [eegEnabled, setEegEnabled] = useState<boolean>(true);
  const [equity, setEquity] = useState<number>(10000000);

  // COD date bounds are pinned at mount time to keep the range stable during a session
  const today = useMemo(() => new Date(), []);
  const minCodDate = useMemo(() => toISODate(today), [today]);
  const maxCodDate = useMemo(() => toISODate(addYears(today, 5)), [today]);
  const [codDate, setCodDate] = useState<string>(minCodDate);

  const [email, setEmail] = useState<string>("");

  // UI label still says "Precise calculation", but backend expects fast_mode.
  // fastMode=true => returns immediately; fastMode=false => queue report via email.
  const [fastMode, setFastMode] = useState<boolean>(true); // default: fast

  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  // Normalized per-year output table for the UI
  const [resultTable, setResultTable] = useState<ResultRow[]>([]);
  const [modalOpen, setModalOpen] = useState<boolean>(false);

  // Full backend response (for KPI cards and extra metadata)
  const [resultMeta, setResultMeta] = useState<any | null>(null);

  // Lightweight in-app navigation (keeps it simple without adding a router)
  const [page, setPage] = useState<"main" | "remarks">("main");

  // backend selector (remembered)
  /*const [apiTarget, setApiTarget] = useState<"render" | "local">(() => {
    const saved = localStorage.getItem("apiTarget");
    return saved === "local" ? "local" : "render";
  });

  useEffect(() => {
    localStorage.setItem("apiTarget", apiTarget);
  }, [apiTarget]);

  const API_BASE = apiTarget === "local" ? LOCAL_API_BASE : RENDER_API_BASE;*/

  /**
   * Map user-facing turbine model to the backend enum.
   * Keep in sync with the backend model list.
   */
  const turbineTypeIdFromModel = (model: string): 0 | 1 | 2 => {
    switch (model) {
      case "ENERCON E-160 EP5 E3 R1":
        return 0;
      case "Nordex N149/5.X":
        return 2;
      case "Vestas V136-4.5 MW":
        return 1;
      default:
        // Safe fallback: pick a valid id rather than rejecting
        return 1;
    }
  };

  // In fast mode email is optional; in precise mode email must be valid
  const emailOk = fastMode || isValidEmail(email);
  const canSubmit = !isSubmitting && emailOk;

  /**
   * Derive chart series from the yearly table.
   * Memoized so UI updates (e.g., typing) don't re-compute series unnecessarily.
   */
  const revenueCosts = useMemo(() => {
    const points = resultTable
      .map((r) => {
        // Some backends return an explicit year_start; prefer it to avoid drift
        const year = yearFromYearStart(r.year_start) ?? r.year;
        const revenue = r.revenue_p50 ?? null;
        const profit = r.profit_p50 ?? null;

        // Costs are presented as a combined series for readability
        const costsRaw = (r.opex ?? 0) + (r.debt_service ?? 0);
        const costs = Number.isFinite(costsRaw) ? costsRaw : null;

        return { year, revenue, costs, profit };
      })
      .filter((p) => typeof p.year === "number");

    return {
      revenueSeries: points.filter((p) => p.revenue !== null).map((p) => ({ x: p.year, y: p.revenue as number })),
      costsSeries: points.filter((p) => p.costs !== null).map((p) => ({ x: p.year, y: p.costs as number })),
      profitSeries: points.filter((p) => p.profit !== null).map((p) => ({ x: p.year, y: p.profit as number })),
    };
  }, [resultTable]);

  /**
   * Submit handler:
   * - In precise mode: queue the report, show confirmation modal immediately
   * - In fast mode: await the response and show the detailed results modal
   */
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return; // prevents double-click duplicates

    // Precise mode requires a valid email; inline UI shows the validation state
    if (!fastMode && !isValidEmail(email)) {
      return;
    }

    setIsSubmitting(true);

    // Request payload must match backend contract
    const payload = {
      latitude,
      longitude,

      n_turbines: turbines,
      hub_height_m: hubHeight,
      turbine_type_id: turbineTypeIdFromModel(turbineType),
      equity_eur: equity,

      tso_id: tsoId,
      eeg_on: Boolean(eegEnabled),
      cod_date: codDate,

      manual_eeg_strike: null,

      fast_mode: fastMode,
      email: !fastMode ? email.trim() : null,
    };

    console.log("calc payload (request)", payload);

    // ✅ PRECISE MODE: send request and immediately show "queued" feedback
    if (!fastMode) {
      // Clear any stale UI state so the modal can't show previous results
      setResultTable([]);
      setResultMeta(null);
      setModalOpen(false);

      // Show the small queued confirmation modal
      setEmailSentModalOpen(true);

      // Fire-and-forget: request runs server-side, user gets results via email
      fetch(`${API_BASE}/api/calc`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
        .then(async (resp) => {
          // Optional: log response for debugging / monitoring
          const data = await resp.json().catch(() => null);
          console.log("precise mode response status", resp.status);
          console.log("precise mode response data", data);
        })
        .catch((err) => {
          console.error("precise mode request failed", err);
        })
        .finally(() => {
          setIsSubmitting(false);
        });

      return; // prevents falling through into the fast-mode flow
    }

    // ✅ FAST MODE: await response and show results in the main modal
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/calc`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await resp.json();

        console.log("response status", resp.status);
        console.log("response data", data);

        if (!resp.ok) {
          // Reset to a clean state on error responses
          setResultTable([]);
          setResultMeta(null);
        } else {
          // Store full response for KPI cards and extra details
          setResultMeta(data);

          // Normalize table payload (supports multiple backend formats)
          const tableRaw = data.yearly_table ?? data.table ?? [];
          const normalized = normalizeYearlyTable(tableRaw);
          setResultTable(normalized);

          // Keep these available for debugging / quick inspection if needed
          const npvMean =
            data?.npv_eur && typeof data.npv_eur === "object" ? data.npv_eur.mean : data?.npv_eur;

          const irrMean =
            data?.irr && typeof data.irr === "object" ? data.irr.mean : data?.irr;

          const irrText =
            irrMean === null || irrMean === undefined || Number.isNaN(Number(irrMean))
              ? "n/a"
              : `${(Number(irrMean) * 100).toFixed(2)}%`;

          void npvMean;
          void irrText;
        }

        setModalOpen(true);
      } catch (err) {
        // Network or parsing errors still open the modal so the user sees "no data" state
        setResultMeta(null);
        setModalOpen(true);
      } finally {
        setIsSubmitting(false);
      }
    })();
  };

  // Simple view switch (remarks/resources page)
  if (page === "remarks") {
    return <RemarksResourcesPage onBack={() => setPage("main")} />;
  }

  /** -------- compute mean stats for modal cards -------- **/

  // KPI card values are derived defensively (avoid NaN/undefined in UI)
  const npvMeanCard =
    resultMeta?.npv_eur && typeof resultMeta.npv_eur === "object"
      ? Number.isFinite(Number(resultMeta.npv_eur.mean))
        ? Number(resultMeta.npv_eur.mean)
        : null
      : null;

  const irrMeanCard =
    resultMeta?.irr && typeof resultMeta.irr === "object"
      ? Number.isFinite(Number(resultMeta.irr.mean))
        ? Number(resultMeta.irr.mean)
        : null
      : null;

  // Total Revenue / Profit mean via final_summary_table (preferred), fall back to stats_table keys
  const totalRevenueMean =
    pickMetricMeanFromFinalSummary(resultMeta?.final_summary_table, "Total revenue (€)") ??
    pickMetricMeanFromStats(resultMeta?.stats_table, "revenue_total_eur");

  const totalProfitMean =
    pickMetricMeanFromFinalSummary(resultMeta?.final_summary_table, "Total profit after OPEX+debt (€)") ??
    pickMetricMeanFromStats(resultMeta?.stats_table, "total_profit_after_opex_and_debt_eur");

  // Centralized classnames keep inputs consistent and easy to tweak
  const inputCls = "w-full border rounded px-2 py-1 text-[11px] sm:text-xs lg:text-sm";
  const labelCls = "block font-medium text-[11px] sm:text-xs lg:text-sm";
  const selectCls = "w-full border rounded px-2 py-1 text-[11px] sm:text-xs lg:text-sm bg-white";
  const btnPrimaryCls = "w-full text-white rounded px-2 py-1.5 text-[11px] sm:text-xs lg:text-sm transition";

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      {/* Map handles pin movement and can optionally update tsoId */}
      <MapView
        latitude={latitude}
        longitude={longitude}
        setLatitude={setLatitude}
        setLongitude={setLongitude}
        setTsoId={setTsoId}
      />

      {/* Settings panel overlay */}
      <form
        onSubmit={handleSubmit}
        className="
          absolute top-2 left-2 z-[1000]
          bg-white/95 backdrop-blur-md shadow-xl rounded-xl border
          w-[min(88vw,19rem)] sm:w-[20rem] lg:w-[22rem]
          max-h-[calc(100vh-1rem)]
          overflow-y-auto
          p-2 sm:p-3 lg:p-4
          space-y-2 sm:space-y-3
          text-[11px] sm:text-xs lg:text-sm
        "
      >
        <h2 className="text-base sm:text-lg font-semibold">Settings</h2>

        {/* backend toggle */}{/*
        ... optional local/render switch UI ...
        */}

        <div>
          <label className={labelCls}>Latitude</label>
          <input
            type="number"
            step="0.01"
            value={latitude}
            onChange={(e) => setLatitude(parseFloat(e.target.value))}
            className={inputCls}
          />
        </div>

        <div>
          <label className={labelCls}>Longitude</label>
          <input
            type="number"
            step="0.0001"
            value={longitude}
            onChange={(e) => setLongitude(parseFloat(e.target.value))}
            className={inputCls}
          />
        </div>

        <div>
          <label className={labelCls}>Number of Turbines</label>
          <input
            type="number"
            min={1}
            max={10}
            value={turbines}
            onChange={(e) => setTurbines(Number(e.target.value))}
            className={inputCls}
          />
        </div>

        <div>
          <label className={labelCls}>Turbine Model</label>
          <select value={turbineType} onChange={(e) => setTurbineType(e.target.value)} className={selectCls}>
            <option value="ENERCON E-160 EP5 E3 R1">ENERCON E-160 EP5 E3 R1</option>
            <option value="Nordex N149/5.X">Nordex N149/5.X</option>
            <option value="Vestas V136-4.5 MW">Vestas V136-4.5 MW</option>
          </select>
        </div>

        <div>
          <label className={labelCls}>Hub Height (m)</label>
          <input
            type="number"
            min={80}
            max={180}
            step={1}
            value={hubHeight}
            onChange={(e) => setHubHeight(Number(e.target.value))}
            className={inputCls}
            placeholder="160"
          />
        </div>

        <div>
          <label className={labelCls}>Inception Date (COD)</label>
          <input
            type="date"
            value={codDate}
            min={minCodDate}
            max={maxCodDate}
            onChange={(e) => setCodDate(e.target.value)}
            className={inputCls}
          />
          <p className="text-[10px] sm:text-xs text-gray-500 mt-0.5">
            Allowed: {minCodDate} → {maxCodDate}
          </p>
        </div>

        <div>
          <label className={labelCls}>Email</label>

          <div className="flex items-center gap-2">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className={`${inputCls} flex-1 ${!fastMode && !isValidEmail(email) ? "border-red-400" : ""}`}
              placeholder="name@company.com"
              required={!fastMode}
              disabled={fastMode}
            />

            {/* Toggle: fast mode (no email) vs precise mode (requires email) */}
            <button
              type="button"
              onClick={() => setFastMode((v) => !v)}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition ${
                !fastMode ? "bg-blue-600" : "bg-gray-300"
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                  !fastMode ? "translate-x-4" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          <div className="mt-0.5 text-[10px] sm:text-xs text-gray-600">
            Precise mode: <span className="font-medium">{!fastMode ? "On" : "Off"}</span>
          </div>

          {/* Inline validation only matters in precise mode */}
          {!fastMode && !isValidEmail(email) && (
            <p className="text-[10px] sm:text-xs text-red-600 mt-0.5">Enter a valid email to run precise mode.</p>
          )}
        </div>

        <div className="flex items-center justify-between">
          <label className={labelCls}>EEG</label>
          <button
            type="button"
            onClick={() => setEegEnabled((v) => !v)}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition ${
              eegEnabled ? "bg-blue-600" : "bg-gray-300"
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                eegEnabled ? "translate-x-4" : "translate-x-1"
              }`}
            />
          </button>
        </div>

        <div>
          <label className={labelCls}>Equity</label>
          <input
            type="number"
            min={1000000}
            max={500000000}
            value={equity}
            onChange={(e) => setEquity(Number.parseInt(e.target.value || "0", 10))}
            className={inputCls}
            placeholder="10000000"
          />
        </div>

        <button
          type="submit"
          disabled={!canSubmit}
          className={`${btnPrimaryCls} ${!canSubmit ? "bg-blue-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"}`}
        >
          {isSubmitting ? "Calculating..." : "Calculate Wind Output"}
        </button>
      </form>

      {/* Quick access to credits/notes without leaving the app */}
      <button
        type="button"
        onClick={() => setPage("remarks")}
        className="absolute bottom-4 right-4 z-[1500] text-xs text-gray-800 bg-white/90 backdrop-blur-md border rounded-full px-3 py-2 shadow hover:bg-white"
      >
        Remarks &amp; Resources
      </button>

      {/* Precise-mode confirmation modal */}
      {emailSentModalOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[2500]">
          <div className="bg-white rounded-xl shadow-xl p-4 w-[28rem] max-w-[90vw] relative">
            <button
              onClick={() => setEmailSentModalOpen(false)}
              className="absolute top-2 right-2 text-gray-500 hover:text-gray-700 font-bold"
              aria-label="Close"
            >
              ✕
            </button>

            <div className="text-sm font-semibold mb-1">Report queued</div>
            <div className="text-sm text-gray-700">An email with a detailed report will arrive in you inbox in a few minutes.</div>

            <div className="mt-3 flex justify-end">
              <button
                type="button"
                onClick={() => setEmailSentModalOpen(false)}
                className="px-3 py-2 rounded border bg-white hover:bg-gray-50 text-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Fast-mode results modal */}
      {modalOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[2000]">
          {/* Scrollable modal so long tables don't overflow the viewport */}
          <div
            className="
              bg-white rounded-xl shadow-xl relative overflow-y-auto
              w-[92vw]
              max-w-[900px] md:max-w-[1000px] xl:max-w-[80rem]
              max-h-[70vh] md:max-h-[72vh] xl:max-h-[90vh]
              p-3 sm:p-5 xl:p-6
            "
          >
            <button onClick={() => setModalOpen(false)} className="absolute top-2 right-2 text-gray-500 hover:text-gray-700 font-bold">
              ✕
            </button>

            <h2 className="text-lg font-semibold mb-2">Wind Output Estimate</h2>

            {/* KPI cards only render when the backend explicitly reports success */}
            {resultMeta?.ok && (
              <div className="grid grid-cols-4 gap-3 mb-4">
                <div className="border rounded-lg p-3 bg-gray-50">
                  <div className="text-xs text-gray-500">NPV (Mean)</div>
                  <div className="text-sm font-semibold">{fmtEURCompact(npvMeanCard)}</div>
                </div>

                <div className="border rounded-lg p-3 bg-gray-50">
                  <div className="text-xs text-gray-500">IRR (Mean)</div>
                  <div className="text-sm font-semibold">{fmtPercent(irrMeanCard, false)}</div>
                </div>

                <div className="border rounded-lg p-3 bg-gray-50">
                  <div className="text-xs text-gray-500">Total Revenue (Mean)</div>
                  <div className="text-sm font-semibold">{fmtEURCompact(totalRevenueMean)}</div>
                </div>

                <div className="border rounded-lg p-3 bg-gray-50">
                  <div className="text-xs text-gray-500">Total Profit (Mean)</div>
                  <div className="text-sm font-semibold">{fmtEURCompact(totalProfitMean)}</div>
                </div>

                <div className="border rounded-lg p-3 bg-gray-50">
                  <div className="text-xs text-gray-500">Discount rate</div>
                  <div className="text-sm font-semibold">
                    {resultMeta?.discount_rate_used !== undefined && resultMeta?.discount_rate_used !== null
                      ? fmtPercent(Number(resultMeta.discount_rate_used), false)
                      : "—"}
                  </div>
                  <div className="text-xs text-gray-600">
                    WACC:{" "}
                    {resultMeta?.wacc !== undefined && resultMeta?.wacc !== null ? fmtPercent(Number(resultMeta.wacc), false) : "—"}
                  </div>
                </div>

                <div className="border rounded-lg p-3 bg-gray-50">
                  <div className="text-xs text-gray-500">EEG strike price</div>
                  <div className="text-sm font-semibold">
                    {resultMeta?.eeg_strike_eur_per_mwh !== null && resultMeta?.eeg_strike_eur_per_mwh !== undefined
                      ? `€${Number(resultMeta.eeg_strike_eur_per_mwh).toFixed(2)}/MWh`
                      : "EEG off"}
                  </div>
                  <div className="text-xs text-gray-600">
                    Uplift:{" "}
                    {resultMeta?.eeg_uplift_total_eur_mean !== undefined && resultMeta?.eeg_uplift_total_eur_mean !== null
                      ? fmtEURCompact(Number(resultMeta.eeg_uplift_total_eur_mean))
                      : "—"}
                  </div>
                </div>

                <div className="border rounded-lg p-3 bg-gray-50">
                  <div className="text-xs text-gray-500">CAPEX</div>
                  <div className="text-sm font-semibold">{resultMeta?.capex_total_eur ? fmtEURCompact(Number(resultMeta.capex_total_eur)) : "—"}</div>
                  <div className="text-xs text-gray-600">Park: {resultMeta?.park_mw ?? "—"} MW</div>
                </div>

                <div className="border rounded-lg p-3 bg-gray-50">
                  <div className="text-xs text-gray-500">Capital structure</div>
                  <div className="text-xs text-gray-700">Equity: {resultMeta?.equity_eur ? fmtEURCompact(Number(resultMeta.equity_eur)) : "—"}</div>
                  <div className="text-xs text-gray-700">Debt: {resultMeta?.debt_eur ? fmtEURCompact(Number(resultMeta.debt_eur)) : "—"}</div>
                  <div className="text-xs text-gray-700">
                    Equity share:{" "}
                    {resultMeta?.equity_share !== undefined && resultMeta?.equity_share !== null
                      ? fmtPercent(Number(resultMeta.equity_share), false)
                      : "—"}
                  </div>
                </div>
              </div>
            )}

            {/* Detailed yearly table + chart */}
            {resultTable.length > 0 && (
              <div className="space-y-4">
                <div className="border rounded overflow-auto max-h-[55vh]">
                  <table className="w-full text-xs border-collapse">
                    <thead className="sticky top-0 bg-white border-b">
                      <tr>
                        <th className="p-2 text-left">Metric</th>
                        {resultTable.map((r) => (
                          <th key={r.year} className="p-2 text-right">
                            {r.year}
                          </th>
                        ))}
                      </tr>
                    </thead>

                    <tbody>
                      {metricRows.map((m) => (
                        <tr key={m.key as string} className="border-b last:border-b-0">
                          <td className="p-2 font-medium">{m.label}</td>
                          {resultTable.map((r) => (
                            <td key={`${m.key}-${r.year}`} className="p-2 text-right">
                              {formatCell(m.key as string, r)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <MiniLineChart
                  title="Revenue vs Costs vs Profit (P50) over time"
                  seriesA={revenueCosts.revenueSeries}
                  seriesB={revenueCosts.costsSeries}
                  seriesC={revenueCosts.profitSeries}
                  labelA="Revenue (P50)"
                  labelB="Costs (OPEX+Debt)"
                  labelC="Profit (P50)"
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

