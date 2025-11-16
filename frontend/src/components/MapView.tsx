import { MapContainer, TileLayer, Marker, Popup, useMapEvents, Rectangle } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import HeatmapLayer from "./HeatmapLayer";
import * as turf from "@turf/turf";
import USGeoJSON from "../data/us_map.json";

import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});
const US_BOUNDS: L.LatLngBoundsExpression = [
  [24.396308, -124.848974], 
  [49.384358, -66.885444],
];

type Props = {
  latitude: number;
  longitude: number;
  setLatitude: (v: number) => void;
  setLongitude: (v: number) => void;
};

function ClickHandler({
  setLatitude,
  setLongitude,
}: {
  setLatitude: (v: number) => void;
  setLongitude: (v: number) => void;
}) {
  useMapEvents({
    click(e) {
      setLatitude(e.latlng.lat);
      setLongitude(e.latlng.lng);
    },
  });
  return null;
}

// Function to interpolate color from green -> red
function getColor(intensity: number): string {
  // intensity 0 = green, 1 = red
  const r = Math.round(255 * intensity);
  const g = Math.round(255 * (1 - intensity));
  const b = 0;
  return `rgb(${r},${g},${b})`;
}

// Check if a point is inside any geometry in the GeometryCollection
function isPointInUS(lat: number, lng: number): boolean {
  const pt = turf.point([lng, lat]);
  for (const geom of USGeoJSON.geometries) {
    if (geom.type === "Polygon" || geom.type === "MultiPolygon") {
      if (turf.booleanPointInPolygon(pt, geom as any)) return true;
    }
  }
  return false;
}

// Generate grid rectangles inside US polygon
function generateGrid(): { bounds: L.LatLngBoundsExpression; intensity: number }[] {
  const grid: { bounds: L.LatLngBoundsExpression; intensity: number }[] = [];
  const [south, west] = US_BOUNDS[0];
  const [north, east] = US_BOUNDS[1];

  const latStep = 2; // adjust for finer grid
  const lngStep = 2;

  for (let lat = south; lat < north; lat += latStep) {
    for (let lng = west; lng < east; lng += lngStep) {
      const center = [lat + latStep / 2, lng + lngStep / 2];
      if (!isPointInUS(center[0], center[1])) continue;

      // Structured intensity (example: gradient from south-west to north-east)
      const intensity = 0.5 + 0.5 * Math.sin(((lat - south) / (north - south)) * Math.PI) * Math.cos(((lng - west) / (east - west)) * Math.PI);

      grid.push({
        bounds: [
          [lat, lng],
          [Math.min(lat + latStep, north), Math.min(lng + lngStep, east)],
        ],
        intensity,
      });
    }
  }
  return grid;
}

export default function MapView({
  latitude,
  longitude,
  setLatitude,
  setLongitude,
}: Props) {
    const grid = generateGrid();

  return (
    <MapContainer
      center={[latitude, longitude]}
      zoom={5}
      minZoom={5}                  // ⛔ can't zoom out farther
      maxBounds={US_BOUNDS}        // ⛔ can't pan outside US
      maxBoundsViscosity={1.0}     // strong lock at borders
      scrollWheelZoom={true}
      className="h-screen w-screen" // full page!
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="&copy; OpenStreetMap contributors"
      />

      <ClickHandler setLatitude={setLatitude} setLongitude={setLongitude} />

      <Marker position={[latitude, longitude]}>
        <Popup>
          Selected location:
          <br />
          {latitude.toFixed(4)}, {longitude.toFixed(4)}
        </Popup>
      </Marker>
      {/* Choropleth rectangles */}
      {grid.map((cell, i) => (
        <Rectangle
          key={i}
          bounds={cell.bounds}
          pathOptions={{
            fillColor: getColor(cell.intensity),
            fillOpacity: 0.6,
            weight: 0, // no border
          }}
        />
      ))}
    </MapContainer>
  );
}
