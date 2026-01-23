# scripts/make_windscore.py
#
# Build a minimal JSON containing only [lat, lon, score] for every ERA5 grid cell in Germany.
# Score is based on long-term mean ws100 computed from u100/v100:
#   ws100 = sqrt(u100^2 + v100^2)
# and scored by ratio to Germany-wide mean using bins:
#   0: ratio < 0.85
#   1: 0.85 <= ratio < 0.95
#   2: 0.95 <= ratio < 1.05
#   3: 1.05 <= ratio < 1.15
#   4: ratio >= 1.15
#
# Input (relative):  data/raw/era5/ERA5_Germany/*.nc
# Output (relative): data/processed/windscore_points.json

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import xarray as xr


# -----------------------
# Config
# -----------------------
U_NAME = "u100"
V_NAME = "v100"

# score bins for ratio (cell_mean / germany_mean)
BINS = [0.85, 0.95, 1.05, 1.15]

OUT_FILENAME = "windscore_points.json"
ROUND_COORDS = 6


# -----------------------
# Path helpers
# -----------------------
def find_project_root(start: Path) -> Path:
    """
    Finds the directory containing:
      data/raw
      data/processed
    (your backend/wind-power-climate-ml folder)
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "data" / "raw").exists() and (p / "data" / "processed").exists():
            return p
    raise RuntimeError(
        "Could not find project root (expected 'data/raw' and 'data/processed'). "
        "Run this script inside 'wind-power-climate-ml'."
    )


def normalize_lons(lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    If lons are 0..360, convert to -180..180 and return sorting order.
    Returns: (lons_sorted, order_idx)
    """
    lons = np.asarray(lons, dtype=float)
    if np.nanmax(lons) > 180:
        lons_new = ((lons + 180) % 360) - 180
        order = np.argsort(lons_new)
        return lons_new[order], order
    order = np.argsort(lons)
    return lons[order], order


# -----------------------
# NetCDF helpers
# -----------------------
def detect_coord_names(ds: xr.Dataset) -> tuple[str, str]:
    if "latitude" in ds.coords and "longitude" in ds.coords:
        return "latitude", "longitude"
    if "lat" in ds.coords and "lon" in ds.coords:
        return "lat", "lon"
    raise KeyError(f"Could not detect lat/lon coords. Coords: {list(ds.coords)}")


def detect_time_dim(da: xr.DataArray) -> str:
    for cand in ("time", "valid_time"):
        if cand in da.dims:
            return cand
    # fallback: take first dim that is not lat/lon
    for d in da.dims:
        if d.lower() not in ("latitude", "longitude", "lat", "lon"):
            return d
    raise KeyError(f"Could not detect time dimension in dims={da.dims}")


def score_from_ratio(ratio: np.ndarray, bins: list[float]) -> np.ndarray:
    # np.digitize -> values 0..len(bins) => 0..4
    return np.digitize(ratio, bins, right=False).astype(np.int8)


# -----------------------
# Main
# -----------------------
def main():
    # Resolve project root (works for script or notebook)
    try:
        start = Path(__file__).resolve()
    except NameError:
        start = Path.cwd()

    project_root = find_project_root(start)
    raw_dir = project_root / "data" / "raw" / "era5" / "ERA5_Germany"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / OUT_FILENAME

    files = sorted(raw_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in: {raw_dir}")

    print("PROJECT_ROOT:", project_root)
    print("RAW_DIR     :", raw_dir)
    print("OUT_JSON    :", out_json)
    print("N_FILES     :", len(files))

    # Determine coords and basic structure from first file
    with xr.open_dataset(files[0], decode_times=False) as ds0:
        if U_NAME not in ds0.data_vars or V_NAME not in ds0.data_vars:
            raise KeyError(
                f"Need variables '{U_NAME}' and '{V_NAME}'. Found: {list(ds0.data_vars)}"
            )

        lat_name, lon_name = detect_coord_names(ds0)
        lats = ds0[lat_name].values
        lons_raw = ds0[lon_name].values

    # Normalize/sort longitudes and record order
    lons, lon_order = normalize_lons(lons_raw)

    sum_ws = None  # (lat, lon)
    cnt_ws = None  # (lat, lon)

    # Process each monthly file
    for i, fp in enumerate(files, start=1):
        print(f"[{i:03d}/{len(files)}] {fp.name}")

        with xr.open_dataset(fp, decode_times=False) as ds:
            # Some ERA5 exports include expver; average it out if present.
            if "expver" in ds.dims:
                ds = ds.mean("expver")

            if U_NAME not in ds.data_vars or V_NAME not in ds.data_vars:
                raise KeyError(f"{fp.name}: missing {U_NAME}/{V_NAME}. Found: {list(ds.data_vars)}")

            # ws100 = sqrt(u100^2 + v100^2)
            u = ds[U_NAME]
            v = ds[V_NAME]
            time_dim = detect_time_dim(u)

            ws = xr.apply_ufunc(np.hypot, u, v)  # hypot(u,v) = sqrt(u^2+v^2)

            # sum/count over time (weights by available hours, handles NaNs)
            ws_sum = ws.sum(dim=time_dim, skipna=True).astype("float64")
            ws_cnt = ws.count(dim=time_dim).astype("float64")

            # Align lon order to normalized/sorted lons
            # (if raw lons are 0..360 or unsorted)
            ws_sum = ws_sum.isel({lon_name: lon_order})
            ws_cnt = ws_cnt.isel({lon_name: lon_order})

            if sum_ws is None:
                sum_ws = ws_sum
                cnt_ws = ws_cnt
            else:
                sum_ws = sum_ws + ws_sum
                cnt_ws = cnt_ws + ws_cnt

    # Long-term mean per grid cell
    mean_ws = sum_ws / cnt_ws  # (lat, lon)

    # Germany-wide mean
    overall_mean = float(mean_ws.mean(dim=(lat_name, lon_name), skipna=True).values)
    if not np.isfinite(overall_mean) or overall_mean <= 0:
        raise RuntimeError(f"Invalid overall_mean={overall_mean}")

    # Ratio + score
    ratio = (mean_ws / overall_mean).astype("float64").values  # numpy (lat, lon)
    scores = score_from_ratio(ratio, BINS)  # numpy (lat, lon), int8

    # Build minimal payload: list of [lat, lon, score]
    points = []
    for i_lat, lat in enumerate(lats):
        lat_f = round(float(lat), ROUND_COORDS)
        for j_lon, lon in enumerate(lons):
            lon_f = round(float(lon), ROUND_COORDS)
            points.append([lat_f, lon_f, int(scores[i_lat, j_lon])])

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, separators=(",", ":"))

    print("\nDone.")
    print("Saved:", out_json)
    print("Grid :", len(lats), "x", len(lons), "(lat x lon)")
    print("N pts:", len(points))
    print("overall_mean_ws100:", overall_mean)


if __name__ == "__main__":
    main()
