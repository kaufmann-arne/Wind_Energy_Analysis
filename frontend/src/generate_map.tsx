import { useMemo, useEffect, useRef } from "react";
import { createStaticHeatmap } from "./utils/generateStaticHeatmap";

export default function App() {
  const heatmapUrl = useMemo(() => createStaticHeatmap(), []);
  const downloadedRef = useRef(false);

  useEffect(() => {
    // Avoid auto-download during SSR/build tools, and avoid double-run in React StrictMode
    if (typeof window === "undefined") return;
    if (downloadedRef.current) return;
    downloadedRef.current = true;

    const a = document.createElement("a");
    a.href = heatmapUrl;
    a.download = "heatmap.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [heatmapUrl]);

  return (
    <div>
      <img src={heatmapUrl} alt="Germany heatmap" />
    </div>
  );
}
