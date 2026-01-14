import { useState } from 'react';
import './App.css';
import MapView from "./components/MapView";

function App() {
  const [latitude, setLatitude] = useState<number>(51.1657); // Center of Germany
  const [longitude, setLongitude] = useState<number>(10.4515);
  const [turbines, setTurbines] = useState<number>(5);
  const [turbineType, setTurbineType] = useState<string>("Vestas V100");

  // NEW: EEG toggle + equity field
  const [eegEnabled, setEegEnabled] = useState<boolean>(true);
  const [equity, setEquity] = useState<number>(0);

  // Modal state
  const [modalOpen, setModalOpen] = useState<boolean>(false);
  const [modalText, setModalText] = useState<string>("");

const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const payload = {
      n_turbines: turbines,
      hub_height_m: 160,          // wire your hub height state later
      turbine_type_id: 1,         // map from turbineType string later
      equity_eur: equity,

      tso_id: 1,                  // add dropdown later if needed
      eeg_on: eegEnabled,
      cod_date: "2026-01-01",      // can be “today month start” later
      manual_eeg_strike: null,

      mwh_constant: 24000          // placeholder until you send real monthly series
    };

    try {
      const resp = await fetch("/api/calc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await resp.json();

      if (!resp.ok) {
        setModalText(`Backend error: ${data?.detail ?? "Unknown error"}`);
      } else {
        setModalText(
          `NPV: €${(data.npv_eur / 1e6).toFixed(1)}M\n` +
          `IRR: ${(data.irr * 100).toFixed(2)}%\n` +
          `Discount rate: ${(data.discount_rate_used * 100).toFixed(2)}%`
        );
      }

      setModalOpen(true);
    } catch (err) {
      setModalText(`Network error: ${String(err)}`);
      setModalOpen(true);
    }
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
      <form
        onSubmit={handleSubmit}
        className="absolute top-4 left-4 bg-white/95 backdrop-blur-md shadow-xl rounded-xl p-6 w-80 z-[1000] border space-y-4"
      >
        <h2 className="text-xl font-semibold">Settings</h2>

        {/* Latitude */}
        <div>
          <label className="text-sm font-medium">Latitude</label>
          <input
            type="number"
            step="0.0001"
            value={latitude}
            onChange={(e) => setLatitude(parseFloat(e.target.value))}
            className="w-full p-2 border rounded"
          />
        </div>

        {/* Longitude */}
        <div>
          <label className="text-sm font-medium">Longitude</label>
          <input
            type="number"
            step="0.0001"
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
            value={turbines}
            onChange={(e) => setTurbines(Number(e.target.value))}
            className="w-full p-2 border rounded"
          />
        </div>

        {/* Turbine Type */}
        <div>
          <label className="text-sm font-medium">Turbine Model</label>
          <select
            value={turbineType}
            onChange={(e) => setTurbineType(e.target.value)}
            className="w-full p-2 border rounded"
          >
            <option value="Vestas V100">Vestas V100</option>
            <option value="Vestas V136">Vestas V136</option>
            <option value="GE 1.5sle">GE 1.5sle</option>
            <option value="Siemens SWT-3.6">Siemens SWT-3.6</option>
          </select>
        </div>

        {/* Hub Height */}
        <div>
          <label className="text-sm font-medium">Hub Height (m)</label>
          <input
            type="number"
            className="w-full p-2 border rounded"
            placeholder="e.g. 120"
          />
        </div>

        {/* NEW: EEG toggle */}
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">EEG</label>
          <button
            type="button"
            onClick={() => setEegEnabled((v) => !v)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition ${
              eegEnabled ? "bg-blue-600" : "bg-gray-300"
            }`}
            aria-pressed={eegEnabled}
            aria-label="Toggle EEG"
          >
            <span
              className={`inline-block h-5 w-5 transform rounded-full bg-white transition ${
                eegEnabled ? "translate-x-5" : "translate-x-1"
              }`}
            />
          </button>
        </div>

        {/* NEW: Equity integer */}
        <div>
          <label className="text-sm font-medium">Equity</label>
          <input
            type="number"
            step={1}
            min={0}
            value={equity}
            onChange={(e) => setEquity(Number.parseInt(e.target.value || "0", 10))}
            className="w-full p-2 border rounded"
            placeholder="e.g. 50000"
          />
        </div>

        <button
          type="submit"
          className="w-full bg-blue-600 hover:bg-blue-700 text-white p-2 rounded"
        >
          Calculate Wind Output
        </button>
      </form>

      {/* MODAL */}
      {modalOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[2000]">
          <div className="bg-white rounded-xl shadow-xl p-6 w-96 relative">
            {/* Close button */}
            <button
              onClick={() => setModalOpen(false)}
              className="absolute top-2 right-2 text-gray-500 hover:text-gray-700 font-bold"
            >
              ✕
            </button>

            {/* Modal content */}
            <h2 className="text-lg font-semibold mb-4">Wind Output Estimate</h2>
            <p className="whitespace-pre-line">{modalText}</p>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;
