import { MapContainer, TileLayer, Marker, Popup, useMapEvents} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

import { ImageOverlay } from "react-leaflet";
import { createStaticHeatmap, HEATMAP_BOUNDS } from "../utils/generateStaticHeatmap";

import { useMemo } from "react";



L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});
import type { LatLngTuple } from "leaflet";

const DE_BOUNDS: [LatLngTuple, LatLngTuple] = [
  [47.2701, 5.8663],
  [55.0581, 15.0419],
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




export default function MapView({
  latitude,
  longitude,
  setLatitude,
  setLongitude,
}: Props) {


  const heatmapUrl = useMemo(() => createStaticHeatmap(), []);


  return (
    <MapContainer
      center={[latitude, longitude]}
      zoom={6.5}
      minZoom={6.5}                  // ⛔ can't zoom out farther
      maxBounds={DE_BOUNDS}
       // ⛔ can't pan outside US
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
        <ImageOverlay
          url={heatmapUrl}
          bounds={HEATMAP_BOUNDS as any}   // or cast to LatLngBoundsExpression
          opacity={0.7}
        />
    </MapContainer>
  );
}
