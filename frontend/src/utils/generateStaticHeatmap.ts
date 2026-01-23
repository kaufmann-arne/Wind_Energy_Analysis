import { heatmapData } from "../data/heatmapData";
import GermanyGeoJSON from "../data/germany.json";

const CLASS_COLORS: Record<number, [number, number, number, number]> = {
  0: [255, 51, 51, 170],
  1: [255, 153, 51, 170],
  2: [255, 255, 51, 170],
  3: [153, 255, 51, 170],
  4: [51, 255, 51, 170],
};

// Germany bounds [southWest, northEast]
type BoundsLL = [[number, number], [number, number]]; // [[south, west], [north, east]]

// Optional clarity types for GeoJSON coordinates
type LngLat = [number, number];
type Ring = LngLat[];
type PolygonCoords = Ring[];
type MultiPolygonCoords = PolygonCoords[];

function computeGeoJsonBoundsLL(): BoundsLL {
  let minLat = Infinity,
    minLng = Infinity,
    maxLat = -Infinity,
    maxLng = -Infinity;

  const geom = GermanyGeoJSON.features?.[0]?.geometry;
  if (!geom) return [[47.27, 5.86], [55.06, 15.04]]; // fallback

  const visit = (lng: number, lat: number) => {
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    if (lng < minLng) minLng = lng;
    if (lng > maxLng) maxLng = lng;
  };

  if (geom.type === "Polygon") {
    (geom.coordinates as unknown as number[][][]).forEach((ring) =>
      ring.forEach(([lng, lat]) => visit(lng, lat))
    );
  } else if (geom.type === "MultiPolygon") {
    (geom.coordinates as MultiPolygonCoords).forEach((poly) =>
      poly.forEach((ring) => ring.forEach(([lng, lat]) => visit(lng, lat)))
    );
  }

  return [[minLat, minLng], [maxLat, maxLng]];
}

export const DE_BOUNDS = computeGeoJsonBoundsLL();

// tiny padding in degrees (keep small now that Mercator is correct)
const padding = 0.0;

export const HEATMAP_BOUNDS: [[number, number], [number, number]] = [
  [DE_BOUNDS[0][0] - padding, DE_BOUNDS[0][1] - padding],
  [DE_BOUNDS[1][0] + padding, DE_BOUNDS[1][1] + padding],
];

const R = 6378137;

function mercatorXY(lat: number, lng: number): [number, number] {
  const d2r = Math.PI / 180;
  const x = R * (lng * d2r);
  const latRad = Math.max(
    Math.min(lat * d2r, Math.PI / 2 - 1e-6),
    -Math.PI / 2 + 1e-6
  );
  const y = R * Math.log(Math.tan(Math.PI / 4 + latRad / 2));
  return [x, y];
}

const knnMajorityClass = (
  x: number,
  y: number,
  projected: { x: number; y: number; classId: number }[],
  k = 5
) => {
  // keep the k best distances
  const best: { d2: number; classId: number }[] = [];

  for (const p of projected) {
    const dx = p.x - x;
    const dy = p.y - y;
    const d2 = dx * dx + dy * dy;

    if (best.length < k) {
      best.push({ d2, classId: p.classId });
      best.sort((a, b) => a.d2 - b.d2);
    } else if (d2 < best[best.length - 1].d2) {
      best[best.length - 1] = { d2, classId: p.classId };
      best.sort((a, b) => a.d2 - b.d2);
    }
  }

  // majority vote among k
  const counts = new Map<number, number>();
  for (const b of best) counts.set(b.classId, (counts.get(b.classId) ?? 0) + 1);

  // break ties by nearest distance among tied classes
  let bestClass = best[0]?.classId ?? 0;
  let bestCount = -1;
  let bestTieDist = Infinity;

  for (const [cls, cnt] of counts.entries()) {
    const nearestDistForClass = best.find((v) => v.classId === cls)?.d2 ?? Infinity;
    if (cnt > bestCount || (cnt === bestCount && nearestDistForClass < bestTieDist)) {
      bestCount = cnt;
      bestClass = cls;
      bestTieDist = nearestDistForClass;
    }
  }

  return bestClass;
};

/**
 * Majority-smooth a 2D class grid (keeps discrete classes, cleans jagged borders).
 * grid: Uint8Array of length w*h with values 0..4
 */
function smoothMajority1D(grid: Uint8Array, w: number, h: number, passes = 3) {
  // ✅ Important: keep these as plain Uint8Array to avoid ArrayBufferLike vs ArrayBuffer generic mismatches
  let cur: Uint8Array = grid;
  let next: Uint8Array = new Uint8Array(cur.length);

  const idx = (x: number, y: number) => y * w + x;

  for (let pass = 0; pass < passes; pass++) {
    next.set(cur);

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const counts = [0, 0, 0, 0, 0];

        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            const yy = y + dy;
            if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
            counts[cur[idx(xx, yy)]]++;
          }
        }

        let best = 0;
        for (let c = 1; c < 5; c++) {
          if (counts[c] > counts[best]) best = c;
        }
        next[idx(x, y)] = best;
      }
    }

    // ✅ safe swap, types stay Uint8Array
    const tmp = cur;
    cur = next;
    next = tmp;
  }

  return cur;
}

export function createStaticHeatmap(): string {
  // --- cache key: bump this when data/colors change ---
  const CACHE_KEY = "heatmap_germany_DEBUG_" + Date.now();
  localStorage.removeItem(CACHE_KEY);
  const cached = localStorage.getItem(CACHE_KEY);
  if (cached) return cached;

  const [south, west] = HEATMAP_BOUNDS[0];
  const [north, east] = HEATMAP_BOUNDS[1];

  // Project bounds to Mercator
  const [minX, minY] = mercatorXY(south, west);
  const [maxX, maxY] = mercatorXY(north, east);

  const spanX = maxX - minX;
  const spanY = maxY - minY;

  // Final output resolution (returned PNG)
  const baseWidth = 2000;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const canvasWidth = Math.round(baseWidth * dpr);
  const canvasHeight = Math.round(canvasWidth * (spanY / spanX));

  // Supersampling factor for smooth edges (keep 1; we’ll smooth via grid+upscale)
  const SS = 1;

  // Final canvas (downscaled)
  const canvas = document.createElement("canvas");
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  const ctx = canvas.getContext("2d")!;

  // Hi-res canvas (where we draw + mask)
  const hi = document.createElement("canvas");
  hi.width = canvasWidth * SS;
  hi.height = canvasHeight * SS;
  const hctx = hi.getContext("2d")!;

  // --- NEW: coarse classification grid (controls speed + smoothness) ---
  // Higher = more detail (slower). 350–600 is a good range.
  const GRID_W = 500;
  const GRID_H = Math.max(1, Math.round(GRID_W * (spanY / spanX)));

  // Pixel mapping in FINAL canvas coordinates (then scaled by SS)
  const toPixel = (lat: number, lng: number) => {
    const [x, y] = mercatorXY(lat, lng);
    const px = ((x - minX) / spanX) * canvasWidth;
    const py = (1 - (y - minY) / spanY) * canvasHeight;
    return [px, py] as const;
  };

  // Pre-project points in HI canvas coordinates
  const projected = heatmapData.map((p) => {
    const [x, y] = toPixel(p.lat, p.lng);
    return { x: x * SS, y: y * SS, classId: p.classId };
  });

  console.log("heatmapData.length =", heatmapData.length);
  console.log("first points =", heatmapData.slice(0, 5));

  // --- 1) Classify onto coarse grid (store class IDs) ---
  const classGrid = new Uint8Array(GRID_W * GRID_H);
  const gidx = (x: number, y: number) => y * GRID_W + x;

  for (let gy = 0; gy < GRID_H; gy++) {
    for (let gx = 0; gx < GRID_W; gx++) {
      // map coarse grid → HI canvas coordinates
      const hx = (gx / (GRID_W - 1)) * hi.width;
      const hy = (gy / (GRID_H - 1)) * hi.height;

      classGrid[gidx(gx, gy)] = knnMajorityClass(hx, hy, projected, 5);
    }
  }

  // --- 2) Smooth the class grid (discrete majority filter) ---
  const smoothed = smoothMajority1D(classGrid, GRID_W, GRID_H, 3);

  // --- 3) Paint the smoothed grid as an image, then upscale to HI canvas ---
  // Create a tiny canvas at grid resolution, paint pixels 1:1, then draw scaled up.
  const gridCanvas = document.createElement("canvas");
  gridCanvas.width = GRID_W;
  gridCanvas.height = GRID_H;
  const gctx = gridCanvas.getContext("2d")!;

  const imageData = gctx.createImageData(GRID_W, GRID_H);
  for (let gy = 0; gy < GRID_H; gy++) {
    for (let gx = 0; gx < GRID_W; gx++) {
      const cls = smoothed[gidx(gx, gy)];
      const [r, g, b, a] = CLASS_COLORS[cls];

      const i = (gy * GRID_W + gx) * 4;
      imageData.data[i] = r;
      imageData.data[i + 1] = g;
      imageData.data[i + 2] = b;
      imageData.data[i + 3] = a;
    }
  }
  gctx.putImageData(imageData, 0, 0);

  // Draw scaled-up onto HI canvas (smooth scaling => smoother borders)
  hctx.imageSmoothingEnabled = true;
  hctx.clearRect(0, 0, hi.width, hi.height);
  hctx.drawImage(gridCanvas, 0, 0, GRID_W, GRID_H, 0, 0, hi.width, hi.height);

  // --- 4) Mask to Germany polygon (do this on HI canvas) ---
  hctx.globalCompositeOperation = "destination-in";
  hctx.beginPath();

  const geometry = GermanyGeoJSON.features?.[0]?.geometry;
  if (geometry) {
    const drawRing = (ring: number[][]) => {
      ring.forEach(([lng, lat], i) => {
        const [x, y] = toPixel(lat, lng); // GeoJSON is [lng, lat]
        const hx = x * SS;
        const hy = y * SS;
        if (i === 0) hctx.moveTo(hx, hy);
        else hctx.lineTo(hx, hy);
      });
      hctx.closePath();
    };

    if (geometry.type === "Polygon") {
      (geometry.coordinates as unknown as number[][][]).forEach(drawRing);
    } else if (geometry.type === "MultiPolygon") {
      (geometry.coordinates as number[][][][]).forEach((poly) => poly.forEach(drawRing));
    }

    hctx.fillStyle = "black";
    hctx.fill();
  }

  hctx.globalCompositeOperation = "source-over";

  // --- 5) Downscale smoothly to final canvas ---
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  ctx.drawImage(hi, 0, 0, hi.width, hi.height, 0, 0, canvasWidth, canvasHeight);

  const png = canvas.toDataURL("image/png");

  // Cache so we only compute once per browser
  try {
    localStorage.setItem(CACHE_KEY, png);
  } catch {
    // localStorage might be full; ignore
  }

  return png;
}
