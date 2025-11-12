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
    <div className="flex h-screen">
      <aside className="w-80 bg-gray-100 p-6 border-r border-gray-300">
        <h1 className="text-2xl font-bold mb-6">Wind Simulator</h1>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div>
            <label className="block mb-1 font-medium">Latitude</label>
            <input
              type="number"
              step="0.00001"
              value={latitude}
              onChange={(e) => setLatitude(Number(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2"
            />
          </div>

          <div>
            <label className="block mb-1 font-medium">Longitude</label>
            <input
              type="number"
              step="0.00001"
              value={longitude}
              onChange={(e) => setLongitude(Number(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2"
            />
          </div>

          <div>
            <label className="block mb-1 font-medium">Number of Turbines</label>
            <input
              type="number"
              min={0}
              value={turbines}
              onChange={(e) => setTurbines(Number(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2"
            />
          </div>

          <div>
            <label className="block mb-1 font-medium">Turbine Type</label>
            <select
              value={turbineType}
              onChange={(e) => setTurbineType(e.target.value)}
              className="w-full border border-gray-300 rounded px-3 py-2"
            >
              <option value="Type A">Type A</option>
              <option value="Type B">Type B</option>
              <option value="Type C">Type C</option>
            </select>
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition"
          >
            Calculate
          </button>
        </form>

        <div className="mt-6 text-sm text-gray-600">
          <p>Click on the map to pick a location.</p>
        </div>
      </aside>

      <main className="flex-1 bg-gray-50 p-6">
        <div className="w-full h-full border-2 border-gray-300 rounded overflow-hidden">
          <MapView
            latitude={latitude}
            longitude={longitude}
            setLatitude={setLatitude}
            setLongitude={setLongitude}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
