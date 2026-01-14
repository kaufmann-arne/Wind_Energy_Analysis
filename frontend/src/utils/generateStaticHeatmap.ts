import { heatmapData } from "../data/heatmapData";
import GermanyGeoJSON from "../data/germany_map.json";

// Germany bounds [southWest, northEast]
export const DE_BOUNDS: [[number, number], [number, number]] = [
  [47.2701, 5.8663],  // SW  (lat, lng)
  [55.0581, 15.0419], // NE
];

// Padding in degrees (positive expands the overlay a bit beyond the bbox)
const padding = 0.5;

const overlayBounds: [[number, number], [number, number]] = [
  [DE_BOUNDS[0][0] - padding, DE_BOUNDS[0][1] - padding], // SW
  [DE_BOUNDS[1][0] + padding, DE_BOUNDS[1][1] + padding], // NE
];

export function createStaticHeatmap(): string {
  const [south, west] = overlayBounds[0];
  const [north, east] = overlayBounds[1];
  const latSpan = north - south;
  const lngSpan = east - west;

  // Adjust canvas to match geographic aspect ratio
  const canvasWidth = 1200;
  const canvasHeight = Math.round(canvasWidth * (latSpan / lngSpan));
  const canvas = document.createElement("canvas");
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  const ctx = canvas.getContext("2d")!;

  // --- 1. Draw heatmap points ---
  const intensityMap = Array.from({ length: canvas.height }, () =>
    Array(canvas.width).fill(0)
  );

  heatmapData.forEach((point) => {
    const xPoint = ((point.lng - west) / (east - west)) * canvas.width;
    const yPoint = ((north - point.lat) / (north - south)) * canvas.height;

    const radius = 40;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const px = Math.floor(xPoint + dx);
        const py = Math.floor(yPoint + dy);
        if (px >= 0 && px < canvas.width && py >= 0 && py < canvas.height) {
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist <= radius) {
            intensityMap[py][px] += point.value * (1 - dist / radius);
          }
        }
      }
    }
  });

  let maxIntensity = 0;
  intensityMap.forEach((row) =>
    row.forEach((val) => {
      if (val > maxIntensity) maxIntensity = val;
    })
  );

  const imageData = ctx.createImageData(canvas.width, canvas.height);
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const i = (y * canvas.width + x) * 4;
      const norm = maxIntensity ? intensityMap[y][x] / maxIntensity : 0;

      let r, g, b;
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
      imageData.data[i + 3] = 180; // opacity
    }
  }

  ctx.putImageData(imageData, 0, 0);

  // --- 2. Mask to Germany polygon (GeoJSON FeatureCollection) ---
  ctx.globalCompositeOperation = "destination-in";
  ctx.beginPath();

  const geometry = GermanyGeoJSON.features?.[0]?.geometry;

  if (!geometry) {
    // If geometry is missing, don't mask (helps debugging instead of silently failing)
    ctx.globalCompositeOperation = "source-over";
    return canvas.toDataURL("image/png");
  }

  const drawRing = (ring: number[][]) => {
    ring.forEach(([lng, lat], i) => {
      const x = ((lng - west) / (east - west)) * canvas.width;
      const y = ((north - lat) / (north - south)) * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
  };

  if (geometry.type === "Polygon") {
    // coordinates: [ring1, ring2, ...] where ring1 is exterior, others are holes
    (geometry.coordinates as number[][][]).forEach(drawRing);
  } else if (geometry.type === "MultiPolygon") {
    // coordinates: [[ring1, ring2, ...], [ring1, ...], ...]
    (geometry.coordinates as number[][][][]).forEach((polygon) => {
      polygon.forEach(drawRing);
    });
  }

  ctx.fillStyle = "black";
  ctx.fill();
  ctx.globalCompositeOperation = "source-over";



  return canvas.toDataURL("image/png");
}
