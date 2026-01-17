import { heatmapData } from "../data/heatmapData";
import GermanyGeoJSON from "../data/germany.json";

// Germany bounds [southWest, northEast]
type BoundsLL = [[number, number], [number, number]]; // [[south, west], [north, east]]

function computeGeoJsonBoundsLL(): BoundsLL {
  let minLat = Infinity, minLng = Infinity, maxLat = -Infinity, maxLng = -Infinity;

  const geom = GermanyGeoJSON.features?.[0]?.geometry;
  if (!geom) return [[47.27, 5.86], [55.06, 15.04]]; // fallback

  const visit = (lng: number, lat: number) => {
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    if (lng < minLng) minLng = lng;
    if (lng > maxLng) maxLng = lng;
  };

  if (geom.type === "Polygon") {
    (geom.coordinates as number[][][]).forEach(ring =>
      ring.forEach(([lng, lat]) => visit(lng, lat))
    );
  } else if (geom.type === "MultiPolygon") {
    (geom.coordinates as number[][][][]).forEach(poly =>
      poly.forEach(ring => ring.forEach(([lng, lat]) => visit(lng, lat)))
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
  const latRad = Math.max(Math.min(lat * d2r, Math.PI / 2 - 1e-6), -Math.PI / 2 + 1e-6);
  const y = R * Math.log(Math.tan(Math.PI / 4 + latRad / 2));
  return [x, y];
}

export function createStaticHeatmap(): string {
  const [south, west] = HEATMAP_BOUNDS[0];
  const [north, east] = HEATMAP_BOUNDS[1];

  // Project bounds to Mercator
  const [minX, minY] = mercatorXY(south, west);
  const [maxX, maxY] = mercatorXY(north, east);

  const spanX = maxX - minX;
  const spanY = maxY - minY;

  // Canvas size based on Mercator aspect ratio (NOT lat/lng ratio)
  const baseWidth = 2000; // 2000–3000 usually looks much better
  const dpr = Math.min(window.devicePixelRatio || 1, 2); // cap at 2 to avoid huge memory
  const canvasWidth = Math.round(baseWidth * dpr);

  const canvasHeight = Math.round(canvasWidth * (spanY / spanX));

  const canvas = document.createElement("canvas");
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  const ctx = canvas.getContext("2d")!;

  const toPixel = (lat: number, lng: number) => {
    const [x, y] = mercatorXY(lat, lng);
    const px = ((x - minX) / spanX) * canvasWidth;
    const py = (1 - (y - minY) / spanY) * canvasHeight; // invert Y for canvas
    return [px, py] as const;
  };

  // --- 1) Heatmap ---
  const intensityMap = Array.from({ length: canvasHeight }, () =>
    Array(canvasWidth).fill(0)
  );

  heatmapData.forEach((p) => {
    const [xPoint, yPoint] = toPixel(p.lat, p.lng);

    const radius = 40;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const px = Math.floor(xPoint + dx);
        const py = Math.floor(yPoint + dy);
        if (px >= 0 && px < canvasWidth && py >= 0 && py < canvasHeight) {
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist <= radius) {
            intensityMap[py][px] += p.value * (1 - dist / radius);
          }
        }
      }
    }
  });

  let maxIntensity = 0;
  for (let y = 0; y < canvasHeight; y++) {
    for (let x = 0; x < canvasWidth; x++) {
      if (intensityMap[y][x] > maxIntensity) maxIntensity = intensityMap[y][x];
    }
  }

  const imageData = ctx.createImageData(canvasWidth, canvasHeight);
  for (let y = 0; y < canvasHeight; y++) {
    for (let x = 0; x < canvasWidth; x++) {
      const i = (y * canvasWidth + x) * 4;
      const norm = maxIntensity ? intensityMap[y][x] / maxIntensity : 0;

      let r: number, g: number, b: number;
      if (norm < 0.5) {
        const t = norm / 0.5;
        r = Math.round(255 * t);
        g = 255;
        b = 0;
      } else {
        const t = (norm - 0.5) / 0.5;
        r = 255;
        g = Math.round(255 * (1 - t));
        b = 0;
      }

      imageData.data[i] = r;
      imageData.data[i + 1] = g;
      imageData.data[i + 2] = b;
      imageData.data[i + 3] = 180;
    }
  }
  ctx.putImageData(imageData, 0, 0);

  // --- 2) Mask to Germany polygon (GeoJSON) ---
  ctx.globalCompositeOperation = "destination-in";
  ctx.beginPath();

  const geometry = GermanyGeoJSON.features?.[0]?.geometry;
  if (!geometry) {
    ctx.globalCompositeOperation = "source-over";
    return canvas.toDataURL("image/png");
  }

  const drawRing = (ring: number[][]) => {
    ring.forEach(([lng, lat], i) => {
      const [x, y] = toPixel(lat, lng); // GeoJSON ring is [lng, lat]
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
  };

  if (geometry.type === "Polygon") {
    (geometry.coordinates as number[][][]).forEach(drawRing);
  } else if (geometry.type === "MultiPolygon") {
    (geometry.coordinates as number[][][][]).forEach((poly) => poly.forEach(drawRing));
  }

  ctx.fillStyle = "black";
  ctx.fill();
  ctx.globalCompositeOperation = "source-over";

  return canvas.toDataURL("image/png");
}
