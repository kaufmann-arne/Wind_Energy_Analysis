import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

type Props = {
  latitude: number;
  longitude: number;
  setLatitude: (v: number) => void;
  setLongitude: (v: number) => void;
};

function ClickHandler({ setLatitude, setLongitude }: { setLatitude: (v:number)=>void, setLongitude: (v:number)=>void }) {
  useMapEvents({
    click(e) {
      setLatitude(e.latlng.lat);
      setLongitude(e.latlng.lng);
    },
  });
  return null;
}

export default function MapView({ latitude, longitude, setLatitude, setLongitude }: Props) {
  return (
    <MapContainer
      center={[latitude, longitude]}
      zoom={4}
      scrollWheelZoom={true}
      className="h-full w-full rounded"
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="&copy; OpenStreetMap contributors"
      />

      {/* Click handler */}
      <ClickHandler setLatitude={setLatitude} setLongitude={setLongitude} />

      <Marker position={[latitude, longitude]}>
        <Popup>
          Selected location: <br />
          {latitude.toFixed(4)}, {longitude.toFixed(4)}
        </Popup>
      </Marker>
    </MapContainer>
  );
}
