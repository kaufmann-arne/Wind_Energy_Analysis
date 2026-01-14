import { MapContainer, TileLayer, Marker, Popup, useMapEvents} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

import { ImageOverlay } from "react-leaflet";
import { createStaticHeatmap } from "../utils/generateStaticHeatmap";


L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});
import type { LatLngTuple } from "leaflet";

const US_BOUNDS: [LatLngTuple, LatLngTuple] = [
  [47.2701, 5.8663],   // SW  (south-west)
  [55.0581, 15.0419]  // NE (north-east)
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

  const heatmapUrl = createStaticHeatmap();

  return (
    <MapContainer
      center={[latitude, longitude]}
      zoom={6.5}
      minZoom={6.5}                  // ⛔ can't zoom out farther
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
      <ImageOverlay
  url={heatmapUrl}
  bounds={US_BOUNDS}
  opacity={0.7}
/>
    </MapContainer>
  );
}
