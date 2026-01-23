import { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import germanyGeo from "../data/germany.json";

import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

import { ImageOverlay, GeoJSON } from "react-leaflet";
import tsoZones from "../data/tso_zones.json";

import type { LatLngTuple } from "leaflet";

L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});

const DE_BOUNDS: [LatLngTuple, LatLngTuple] = [
  [47.2701, 5.8663],
  [55.0581, 15.0419],
];

type Props = {
  latitude: number;
  longitude: number;
  setLatitude: (v: number) => void;
  setLongitude: (v: number) => void;
  setTsoId: (v: number) => void;
};

type LngLat = [number, number];

// Ray-casting point-in-ring (ring is [[lng,lat],...])
function pointInRing(lng: number, lat: number, ring: LngLat[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0],
      yi = ring[i][1];
    const xj = ring[j][0],
      yj = ring[j][1];

    const intersect =
      yi > lat !== yj > lat && lng < ((xj - xi) * (lat - yi)) / (yj - yi + 0.0) + xi;

    if (intersect) inside = !inside;
  }
  return inside;
}

// Polygon: first ring = outer, subsequent rings = holes
function pointInPolygon(lng: number, lat: number, rings: LngLat[][]): boolean {
  if (!rings?.length) return false;

  // must be inside outer ring
  if (!pointInRing(lng, lat, rings[0])) return false;

  // must NOT be inside any hole
  for (let r = 1; r < rings.length; r++) {
    if (pointInRing(lng, lat, rings[r])) return false;
  }
  return true;
}

function isInsideGermany(lat: number, lng: number): boolean {
  const feature = (germanyGeo as any)?.features?.[0];
  const geom = feature?.geometry;
  if (!geom) return false;

  if (geom.type === "Polygon") {
    return pointInPolygon(lng, lat, geom.coordinates as LngLat[][]);
  }

  if (geom.type === "MultiPolygon") {
    const polys = geom.coordinates as LngLat[][][];
    for (const rings of polys) {
      if (pointInPolygon(lng, lat, rings)) return true;
    }
    return false;
  }

  return false;
}

function ClickHandler({
  setLatitude,
  setLongitude,
}: {
  setLatitude: (v: number) => void;
  setLongitude: (v: number) => void;
}) {
  useMapEvents({
    click(e) {
      const lat = e.latlng.lat;
      const lng = e.latlng.lng;

      // ✅ Exact border check
      if (!isInsideGermany(lat, lng)) return;

      setLatitude(lat);
      setLongitude(lng);
    },
  });
  return null;
}

function LayerToggleControl({
  showHeatmap,
  setShowHeatmap,
  showTso,
  setShowTso,
}: {
  showHeatmap: boolean;
  setShowHeatmap: React.Dispatch<React.SetStateAction<boolean>>;
  showTso: boolean;
  setShowTso: React.Dispatch<React.SetStateAction<boolean>>;
}) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    // Leaflet-approved: stop clicks/scroll from reaching the map
    L.DomEvent.disableClickPropagation(ref.current);
    L.DomEvent.disableScrollPropagation(ref.current);

    // Prevent drag start ("map sticks to mouse") from mousedown on the control
    L.DomEvent.on(ref.current, "mousedown", L.DomEvent.stopPropagation);
    L.DomEvent.on(ref.current, "mousedown", L.DomEvent.preventDefault);

    // Mobile: prevent touch drag
    L.DomEvent.on(ref.current, "touchstart", L.DomEvent.stopPropagation);
    L.DomEvent.on(ref.current, "touchstart", L.DomEvent.preventDefault);

    return () => {
      if (!ref.current) return;
      L.DomEvent.off(ref.current, "mousedown", L.DomEvent.stopPropagation);
      L.DomEvent.off(ref.current, "mousedown", L.DomEvent.preventDefault);
      L.DomEvent.off(ref.current, "touchstart", L.DomEvent.stopPropagation);
      L.DomEvent.off(ref.current, "touchstart", L.DomEvent.preventDefault);
    };
  }, []);

  return (
    <div className="leaflet-top leaflet-right">
      <div
        ref={ref}
        className="leaflet-control m-3 bg-white/95 backdrop-blur-md shadow-lg rounded-xl p-3 border space-y-3 w-52"
      >
        <div className="text-sm font-semibold">Layers</div>

        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Heatmap</span>
          <button
            type="button"
            onClick={() => setShowHeatmap((v) => !v)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition ${
              showHeatmap ? "bg-blue-600" : "bg-gray-300"
            }`}
            aria-pressed={showHeatmap}
          >
            <span
              className={`inline-block h-5 w-5 transform rounded-full bg-white transition ${
                showHeatmap ? "translate-x-5" : "translate-x-1"
              }`}
            />
          </button>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">TSO zones</span>
          <button
            type="button"
            onClick={() => setShowTso((v) => !v)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition ${
              showTso ? "bg-blue-600" : "bg-gray-300"
            }`}
            aria-pressed={showTso}
          >
            <span
              className={`inline-block h-5 w-5 transform rounded-full bg-white transition ${
                showTso ? "translate-x-5" : "translate-x-1"
              }`}
            />
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Creates a dedicated pane that always renders above overlays like the heatmap.
 * This keeps the Germany border line visible even when the heatmap is on.
 */
function EnsureBorderPane({ paneName = "borderPane", zIndex = 650 }: { paneName?: string; zIndex?: number }) {
  const map = useMapEvents({});
  useEffect(() => {
    if (!map) return;

    const existing = map.getPane(paneName);
    if (!existing) {
      map.createPane(paneName);
    }
    const pane = map.getPane(paneName);
    if (pane) {
      pane.style.zIndex = String(zIndex);
      pane.style.pointerEvents = "none"; // never blocks clicks
    }
  }, [map, paneName, zIndex]);

  return null;
}

export default function MapView({ latitude, longitude, setLatitude, setLongitude, setTsoId }: Props) {
  const defaultZoom = window.innerWidth < 1281 ? 6.0 : 7.0;
  const heatmapUrl = "/heatmap.png";

  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showTso, setShowTso] = useState(true);

  const tsoStyle = (feature: any) => {
    const tso = feature?.properties?.tso;

    const colorByTso: Record<string, string> = {
      "50Hertz": "#FF69B4",
      "amprion": "#AA78FF",
      "TenneT": "#40E0D0",
      "TransnetBW": "#4682FF"
    };

    return {
      color: colorByTso[tso] ?? "#111827",
      weight: 2,
      fillColor: colorByTso[tso] ?? "#111827",
      fillOpacity: 0.2,
    };
  };

  // ALWAYS-ON: Germany border (black outline) style
  const germanyBorderStyle = () => {
    return {
      color: "#000000", // dark black border
      weight: 3,
      opacity: 1,
      fillOpacity: 0, // outline only
    };
  };

  const normalizeTso = (v: unknown) =>
    String(v ?? "")
      .trim()
      .toLowerCase()
      .replace(/\s+/g, "")
      .replace(/[^a-z0-9]/g, "");

  const TSO_KEY_TO_ID: Record<string, number> = {
    "50hertz": 0,
    "amprion": 1,
    "tennet": 2,
    "transnetbw": 3,
    "transnetbwgmbh": 3,
    "tennettsogmbh": 2,
    "50hertztransmission": 0,
  };

  const onEachTsoFeature = (feature: any, layer: any) => {
    layer.on("click", () => {
      const raw = feature?.properties?.tso;
      const key = normalizeTso(raw);
      const mapped = TSO_KEY_TO_ID[key];

      if (mapped === undefined) {
        console.warn("Unknown TSO value:", raw, "normalized:", key);
        return;
      }
      setTsoId(mapped);
    });
  };

  return (
    
    <MapContainer
      center={[latitude, longitude]}
      zoom={defaultZoom}
      minZoom={6.0}
      maxBounds={DE_BOUNDS}
      maxBoundsViscosity={1.0}
      scrollWheelZoom={true}
      className="h-screen w-screen"
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="&copy; OpenStreetMap contributors"
      />

      {/* Top-right layer toggles */}
      <LayerToggleControl
        showHeatmap={showHeatmap}
        setShowHeatmap={setShowHeatmap}
        showTso={showTso}
        setShowTso={setShowTso}
      />

      {/* Heatmap overlay */}
      {showHeatmap && <ImageOverlay url={heatmapUrl} bounds={DE_BOUNDS as any} opacity={0.7} />}

      {/* TSO overlay */}
      {showTso && tsoZones && (
        <GeoJSON data={tsoZones as any} style={tsoStyle as any} onEachFeature={onEachTsoFeature as any} />
      )}

      {/* ALWAYS-ON: ensure pane exists above overlays */}
      <EnsureBorderPane paneName="borderPane" zIndex={700} />

      {/* ALWAYS-ON: Germany outline on a high z-index pane */}
      <GeoJSON data={germanyGeo as any} style={germanyBorderStyle as any} pane="borderPane" />

      <ClickHandler setLatitude={setLatitude} setLongitude={setLongitude} />

      <Marker position={[latitude, longitude]}>
        <Popup>
          Selected location:
          <br />
          {latitude.toFixed(4)}, {longitude.toFixed(4)}
        </Popup>
      </Marker>
    </MapContainer>
  );
}
