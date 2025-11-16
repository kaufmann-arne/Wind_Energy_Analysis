import { useState } from 'react';
import './App.css';  // Ensure this imports your index.css with @import "tailwindcss";
import MapView from "./components/MapView";

function App() {
  const [latitude, setLatitude] = useState<number>(39.8283); // geographic center of contiguous US
  const [longitude, setLongitude] = useState<number>(-98.5795);
  const [turbines, setTurbines] = useState<number>(5);
  const [turbineType, setTurbineType] = useState<string>("Type A");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // you can call backend API here
    console.log({ latitude, longitude, turbines, turbineType });
  };

return (
  <div className="relative h-screen w-screen overflow-hidden">


    {/* MAP */}
    <MapView
      latitude={latitude}
      longitude={longitude}
      setLatitude={setLatitude}
      setLongitude={setLongitude}
    />

    {/* FLOATING SIDEBAR */}
    <div className="absolute top-4 left-4 bg-white/95 backdrop-blur-md shadow-xl rounded-xl p-6 w-80 z-[1000] border space-y-4">
      <h2 className="text-xl font-semibold">Settings</h2>

      {/* Lat / Lng */}
      <div>
        <label className="text-sm font-medium">Latitude</label>
        <input
          type="number"
          value={latitude}
          onChange={(e) => setLatitude(parseFloat(e.target.value))}
          className="w-full p-2 border rounded"
        />
      </div>

      <div>
        <label className="text-sm font-medium">Longitude</label>
        <input
          type="number"
          value={longitude}
          onChange={(e) => setLongitude(parseFloat(e.target.value))}
          className="w-full p-2 border rounded"
        />
      </div>

      {/* Turbine Count */}
      <div>
        <label className="text-sm font-medium">Number of Turbines</label>
        <input
          type="number"
          min={1}
          className="w-full p-2 border rounded"
        />
      </div>

      {/* Turbine Type */}
      <div>
        <label className="text-sm font-medium">Turbine Model</label>
        <select className="w-full p-2 border rounded">
          <option>Vestas V100</option>
          <option>Vestas V136</option>
          <option>GE 1.5sle</option>
          <option>Siemens SWT-3.6</option>
        </select>
      </div>

      {/* Placeholder for future */}
      <div>
        <label className="text-sm font-medium">Hub Height (m)</label>
        <input type="number" className="w-full p-2 border rounded" />
      </div>

      <button className="w-full bg-blue-600 hover:bg-blue-700 text-white p-2 rounded">
        Calculate Wind Output
      </button>
    </div>
  </div>
);


}

export default App;
