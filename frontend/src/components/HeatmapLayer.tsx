// src/components/HeatmapLayer.tsx
import { useMap } from "react-leaflet";
import { useEffect } from "react";
import L from "leaflet";
import "leaflet.heat";

type HeatmapProps = {
  points: Array<[number, number, number]>; // [lat, lng, intensity 0-1]
};

export default function HeatmapLayer({ points }: HeatmapProps) {
  const map = useMap();

  useEffect(() => {
    if (!map) return;

    const zoom = map.getZoom();
    const scaledPoints = points.map(([lat, lng, intensity]) => [
      lat,
      lng,
      intensity / Math.pow(2, zoom / 6) // tweak exponent to taste
    ]);


    const heat = (L as any).heatLayer(scaledPoints, {
      radius: 250,
      blur: 1,
      gradient: {
        0.0: "rgba(0,255,0,0.8)",    // green, more opaque
        0.5: "rgba(255,255,0,0.8)",  // yellow
        1.0: "rgba(255,0,0,0.8)",    // red
      },
      maxZoom: 12,
    }).addTo(map);

    return () => {
      map.removeLayer(heat);
    };
  }, [map, points]);

  return null;
}
