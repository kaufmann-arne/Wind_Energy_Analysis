export const heatmapData = (() => {
  const points: { lat: number; lng: number; value: number }[] = [];

  // US approximate bounds (conterminous 48 states)
  const south = 24.396308;
  const north = 49.384358;
  const west = -124.848974;
  const east = -66.885444;

  const numPoints = 1000;
  const rows = Math.floor(Math.sqrt(numPoints));
  const cols = Math.ceil(numPoints / rows);

  for (let i = 0; i < rows; i++) {
    const lat = south + ((north - south) / (rows - 1)) * i;
    for (let j = 0; j < cols; j++) {
      const lng = west + ((east - west) / (cols - 1)) * j;

      // Random-ish intensity between 0.3 and 1 for variability
      const value = 0.3 + Math.random() * 0.7;

      points.push({ lat, lng, value });

      if (points.length >= numPoints) break;
    }
    if (points.length >= numPoints) break;
  }

  return points;
})();
