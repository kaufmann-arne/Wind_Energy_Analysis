import { useState } from 'react';
import './App.css';
import MapView from "./components/MapView";

function App() {
  const [latitude, setLatitude] = useState<number>(51.1657); // Center of Germany
  const [longitude, setLongitude] = useState<number>(10.4515);
  const [turbines, setTurbines] = useState<number>(5);
  const [turbineType, setTurbineType] = useState<number>(0);
  const [hubHeight, setHubHeight] = useState<number>(160);


  // NEW: EEG toggle + equity field
  const [eegEnabled, setEegEnabled] = useState<boolean>(true);
  const [equity, setEquity] = useState<number>(0);

  type ResultRow = {
    year: number;
    mwh_gross: number | null;
    market_price_electricity: number | null;
    cf: number | null;
    cr: number | null;
    opex: number | null;
    debt_service: number | null;
    profit: number | null;
  };

  const [resultTable, setResultTable] = useState<ResultRow[]>([]);

  // Modal state
  const [modalOpen, setModalOpen] = useState<boolean>(false);
  const [modalText, setModalText] = useState<string>("");

const turbineTypeIdFromModel = (model: string): 0 | 1 | 2 => {
  switch (model) {
    case "Vestas V100":
      return 0; // low wind (example)
    case "Vestas V136":
      return 2; // high wind (example)
    case "GE 1.5sle":
      return 1; // balanced
    default:
      return 1;
  }
};



const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const payload = {
      n_turbines: turbines,
      hub_height_m: hubHeight,
      turbine_type_id: turbineTypeIdFromModel(turbineType),
      equity_eur: equity,
      tso_id: 1,
      eeg_on: eegEnabled,
      cod_date: "2026-01-01",
      manual_eeg_strike: null,
      mwh_constant: 12000,
    };

    console.log("payload being sent", payload);

    try {
      const resp = await fetch("/api/calc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await resp.json();

      if (!resp.ok) {
        setModalText(`Backend error: ${data?.detail ?? "Unknown error"}`);
        setResultTable([]);
      } else {
        setResultTable(data.table ?? []);

        const irr = data.irr;
        const irrText = irr === null ? "n/a" : `${(Number(irr) * 100).toFixed(2)}%`;

        setModalText(
          `NPV: €${(Number(data.npv_eur) / 1e6).toFixed(1)}M\n` +
          `IRR: ${irrText}\n` +
          `Discount rate: ${(Number(data.discount_rate_used) * 100).toFixed(2)}%`
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
          </select>
        </div>

        {/* Hub Height */}
        <div>
          <label className="text-sm font-medium">Hub Height (m)</label>
          <input
            type="number"
            min={1}
            step={1}
            value={hubHeight}
            onChange={(e) => setHubHeight(Number(e.target.value))}
            className="w-full p-2 border rounded"
            placeholder="e.g. 160"
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
          <div className="bg-white rounded-xl shadow-xl p-6 w-[48rem] max-h-[90vh] relative overflow-hidden">
            {/* Close button */}
            <button
              onClick={() => setModalOpen(false)}
              className="absolute top-2 right-2 text-gray-500 hover:text-gray-700 font-bold"
            >
              ✕
            </button>

            {/* Modal content */}
            <h2 className="text-lg font-semibold mb-2">Wind Output Estimate</h2>

            {/* Summary numbers */}
            <pre className="whitespace-pre-wrap text-sm bg-gray-50 p-3 rounded border mb-4">
              {modalText}
            </pre>

            {/* Result table */}
            {resultTable.length > 0 && (
              <div className="border rounded overflow-auto max-h-[55vh]">
                <table className="w-full text-xs border-collapse">
                  <thead className="sticky top-0 bg-white border-b">
                    <tr>
                      <th className="p-2 text-left">Year</th>
                      <th className="p-2 text-right">MWh gross</th>
                      <th className="p-2 text-right">Market €/MWh</th>
                      <th className="p-2 text-right">CF</th>
                      <th className="p-2 text-right">CR</th>
                      <th className="p-2 text-right">OPEX (€)</th>
                      <th className="p-2 text-right">Debt (€)</th>
                      <th className="p-2 text-right">Profit (€)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {resultTable.map((r) => (
                      <tr key={r.year} className="border-b last:border-b-0">
                        <td className="p-2">{r.year}</td>
                        <td className="p-2 text-right">
                          {r.mwh_gross !== null ? r.mwh_gross.toLocaleString() : "—"}
                        </td>
                        <td className="p-2 text-right">
                          {r.market_price_electricity !== null
                            ? r.market_price_electricity.toFixed(2)
                            : "—"}
                        </td>
                        <td className="p-2 text-right">
                          {r.cf !== null ? r.cf.toFixed(3) : "—"}
                        </td>
                        <td className="p-2 text-right">
                          {r.cr !== null ? r.cr.toFixed(3) : "—"}
                        </td>
                        <td className="p-2 text-right">
                          {r.opex !== null ? r.opex.toLocaleString() : "—"}
                        </td>
                        <td className="p-2 text-right">
                          {r.debt_service !== null ? r.debt_service.toLocaleString() : "—"}
                        </td>
                        <td className="p-2 text-right font-medium">
                          {r.profit !== null ? r.profit.toLocaleString() : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
