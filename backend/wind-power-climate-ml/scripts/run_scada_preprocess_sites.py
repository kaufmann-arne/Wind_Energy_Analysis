from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scada.preprocess import preprocess_scada_dataset
from scada.greenbyte import process_site as process_greenbyte_site
from scada.loegtved import (
    split_loegtved_txt_to_turbine_csvs,
    aggregate_loegtved_turbine_hourly,
    aggregate_loegtved_park_hourly,
)

import pandas as pd


def main() -> None:
    PROJECT_ROOT = REPO_ROOT

    turbine_db_path = PROJECT_ROOT / "Data" / "turbine_power_curves.json"
    scada_final_dir = PROJECT_ROOT / "Data" / "Processed" / "Scada_final"
    scada_final_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # A) GREENBYTE: Kelmarsh + Penmanshiel -> park_hourly.csv
    # =========================================================
    kelmarsh_turbines = [
        "Kelmarsh_1",
        "Kelmarsh_2",
        "Kelmarsh_3",
        "Kelmarsh_4",
        "Kelmarsh_5",
        "Kelmarsh_6",
    ]
    process_greenbyte_site(project_root=PROJECT_ROOT, site_name="Kelmarsh", turbine_ids=kelmarsh_turbines)

    penmanshiel_turbines = [
        "Penmanshiel_01",
        "Penmanshiel_02",
        "Penmanshiel_04",
        "Penmanshiel_05",
        "Penmanshiel_06",
        "Penmanshiel_07",
        "Penmanshiel_08",
        "Penmanshiel_09",
        "Penmanshiel_10",
        "Penmanshiel_11",
        "Penmanshiel_12",
        "Penmanshiel_13",
        "Penmanshiel_14",
        "Penmanshiel_15",
    ]
    process_greenbyte_site(project_root=PROJECT_ROOT, site_name="Penmanshiel", turbine_ids=penmanshiel_turbines)

    # =========================================================
    # B) LOEGTVED: TXT -> split -> turbine_hourly -> park_hourly
    # =========================================================
    loegtved_raw = PROJECT_ROOT / "Data" / "Raw" / "Scada" / "Scada_Loegtved" / "V100_3_POWER_WIND_PITCH_ALARM.TXT"
    loegtved_outdir = PROJECT_ROOT / "Data" / "Processed" / "Loegtved_Pre"
    loegtved_outdir.mkdir(parents=True, exist_ok=True)

    turbine_csvs = split_loegtved_txt_to_turbine_csvs(
        input_path=loegtved_raw,
        output_dir=loegtved_outdir,
        timestamp_col="PCTimeStamp",
    )

    hourly_frames = []
    for p in turbine_csvs:
        wtg = p.stem.replace("_SCADA", "")
        h = aggregate_loegtved_turbine_hourly(p, timestamp_col="PCTimeStamp")
        h["turbine"] = wtg
        hourly_frames.append(h)

    hourly_all = pd.concat(hourly_frames, ignore_index=True)
    park_hourly = aggregate_loegtved_park_hourly(
        hourly_all,
        timestamp_col="PCTimeStamp",
        n_turbines_total=3,       
        availability_fraction_threshold=2/3,
    )


    loegtved_park_path = loegtved_outdir / "Loegtved_park_hourly.csv"
    park_hourly.to_csv(loegtved_park_path, index=False, sep=",", date_format="%Y-%m-%d %H:%M:%S")
    print(f"✓ Loegtved park hourly saved: {loegtved_park_path}")

    # =========================================================
    # C) GENERIC PREPROCESS -> scada_<site>_processed.csv
    # =========================================================


    # --- Dundalk (raw)
    dundalk_meta = {
        "Dundalk_T01": {"lat": 53.98352, "lon": -6.391390, "hub_height_m": 65, "turbine_type": "Vestas_V52_850"}
    }
    preprocess_scada_dataset(
        input_dir=PROJECT_ROOT / "Data" / "Raw" / "Scada" / "Scada_Dundalk",
        output_dir=scada_final_dir,
        site_id="Dundalk",
        site_lat=53.98352,
        site_lon=-6.391390,
        timestamp_col="Timestamps",
        target_col="Power",
        windspeed_col="WindSpeed",
        turbine_meta=dundalk_meta,
        turbine_db_path=turbine_db_path,
        max_power=850,
    )

    # --- Kelmarsh (from park_hourly)
    kelmarsh_meta = {
        "Kelmarsh_T01": {"lat": 52.401111, "lon": -0.942778, "hub_height_m": 75, "turbine_type": "Senvion_MM92"}
    }
    preprocess_scada_dataset(
        input_dir=PROJECT_ROOT / "Data" / "Processed" / "Pen_Kel_Pre" / "Kelmarsh" / "Kelmarsh_park_hourly.csv",
        output_dir=scada_final_dir,
        site_id="Kelmarsh",
        site_lat=52.401111,
        site_lon=-0.942778,
        timestamp_col="timestamp",
        target_col="energy_kwh_mean",
        windspeed_col="wind_speed_ms_mean",
        turbine_meta=kelmarsh_meta,
        turbine_db_path=turbine_db_path,
        max_power=2050,
        qc_col="is_available",
    )

    # --- Penmanshiel (from park_hourly)
    penmanshiel_meta = {
        "Penmanshiel_T01": {"lat": 55.903611, "lon": -2.291667, "hub_height_m": 59, "turbine_type": "Senvion_MM82"}
    }
    preprocess_scada_dataset(
        input_dir=PROJECT_ROOT / "Data" / "Processed" / "Pen_Kel_Pre" / "Penmanshiel" / "Penmanshiel_park_hourly.csv",
        output_dir=scada_final_dir,
        site_id="Penmanshiel",
        site_lat=55.903611,
        site_lon=-2.291667,
        timestamp_col="timestamp",
        target_col="energy_kwh_mean",
        windspeed_col="wind_speed_ms_mean",
        turbine_meta=penmanshiel_meta,
        turbine_db_path=turbine_db_path,
        max_power=2050,
        qc_col="is_available",
    )

    # --- Loegtved (from park_hourly)
    loegtved_meta = {
        "Loegtved_T01": {"lat": 55.676444, "lon": 11.274472, "hub_height_m": 95, "turbine_type": "vestas_v100_2_0"}
    }
    preprocess_scada_dataset(
        input_dir=loegtved_park_path,
        output_dir=scada_final_dir,
        site_id="Loegtved",
        site_lat=55.676444,
        site_lon=11.274472,
        timestamp_col="PCTimeStamp",
        target_col="energy_kwh_park",
        windspeed_col="windspeed_mean",
        turbine_meta=loegtved_meta,
        turbine_db_path=turbine_db_path,
        max_power=2000,
        qc_col="is_available",
    )

    print("✓ All SCADA preprocessing completed.")


if __name__ == "__main__":
    main()
