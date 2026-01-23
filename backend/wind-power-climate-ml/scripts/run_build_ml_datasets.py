from __future__ import annotations

import sys
from pathlib import Path

# --- make src importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from ml.dataset_builder import build_ml_datasets_per_turbine


def nc_list(dir_path: Path) -> list[Path]:
    files = sorted(Path(dir_path).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in: {dir_path}")
    return files


def main() -> None:
    PROJECT_ROOT = REPO_ROOT

    ml_out_dir = PROJECT_ROOT / "Data" / "ML"
    ml_out_dir.mkdir(parents=True, exist_ok=True)

    turbine_curve_json = PROJECT_ROOT / "Data" / "turbine_power_curves.json"
    scada_final_dir = PROJECT_ROOT / "Data" / "Processed" / "Scada_final"
    era5_raw_dir = PROJECT_ROOT / "Data" / "Raw" / "Era5"

    # ---------------------------------------------------------
    # Per-site config: matches your notebook calls
    # ---------------------------------------------------------
    sites = [
        dict(
            name="Penmanshiel",
            era5_main=era5_raw_dir / "ERA5_Penmanshiel",
            era5_fsr_zust=era5_raw_dir / "ERA5_Penmanshiel_fsr_zust",
            scada=scada_final_dir / "scada_Penmanshiel_processed.csv",
        ),
        dict(
            name="Kelmarsh",
            era5_main=era5_raw_dir / "ERA5_Kelmarsh",
            era5_fsr_zust=era5_raw_dir / "ERA5_Kelmarsh_fsr_zust",
            scada=scada_final_dir / "scada_Kelmarsh_processed.csv",
        ),
        dict(
            name="Dundalk",
            era5_main=era5_raw_dir / "ERA5_Dundalk",
            era5_fsr_zust=era5_raw_dir / "ERA5_Dundalk_fsr_zust",
            scada=scada_final_dir / "scada_Dundalk_processed.csv",
        ),
        dict(
            name="Loegtved",
            era5_main=era5_raw_dir / "ERA5_Loegtved",
            era5_fsr_zust=era5_raw_dir / "ERA5_Loegtved_fsr_zust",
            scada=scada_final_dir / "scada_Loegtved_processed.csv",
        ),

        # Optional Germany example (all-in-one):
        # dict(
        #     name="Germany_example",
        #     era5_main=era5_raw_dir / "ERA5_Germany",
        #     era5_fsr_zust=era5_raw_dir / "ERA5_Germany",  # <- same dir (all-in-one)
        #     scada=scada_final_dir / "scada_GermanySite_processed.csv",
        # ),
    ]

    # ---------------------------------------------------------
    # Build datasets
    # ---------------------------------------------------------
    for s in sites:
        print(f"\n=== Build ML datasets: {s['name']} ===")

        main_paths = nc_list(Path(s["era5_main"]))
        fsr_paths = nc_list(Path(s["era5_fsr_zust"]))

        datasets = build_ml_datasets_per_turbine(
            era5_main_paths=main_paths,
            era5_fsr_zust_paths=fsr_paths,
            scada_csv_path=Path(s["scada"]),
            turbine_power_curve_json=turbine_curve_json,
            output_dir=ml_out_dir,
            climate_prefix="era5_",
            use_log_target=True,
        )

        # Print what we got (same logic as notebook selecting T01)
        print("Built turbines:", list(datasets.keys())[:10], "..." if len(datasets) > 10 else "")

    # ---------------------------------------------------------
    # Merge all ml_dataset_*.csv -> ml_dataset_ALL_T01.csv
    # (exactly like your notebook)
    # ---------------------------------------------------------
    csvs = sorted(ml_out_dir.glob("ml_dataset_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No ml_dataset_*.csv in {ml_out_dir}")

    dfs = [pd.read_csv(p) for p in csvs]
    df_all = pd.concat(dfs, ignore_index=True)

    out_path = ml_out_dir / "ml_dataset_ALL_T01.csv"
    df_all.to_csv(out_path, index=False)
    print("\n✓ Saved merged dataset at:", out_path.resolve())


if __name__ == "__main__":
    main()
