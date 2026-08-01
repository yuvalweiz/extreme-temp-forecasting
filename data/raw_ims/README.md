# data/raw_ims/ — put your raw IMS export here

This directory is the landing zone for the **raw Israel Meteorological Service (IMS)
export**. It ships empty: IMS data is free but must be obtained from IMS directly
(we do not redistribute it).

## What to download

From the IMS open-data portal (https://ims.data.gov.il — the public archive of
https://ims.gov.il/en), download the country-wide observation archive. The
preprocessing notebook (`preprocessing/01_Preprocessing_Hourly_Data.ipynb`) expects
the portal's standard file names:

| file | content | key columns |
|---|---|---|
| `new_isr_gnd_obs_web.csv` | hourly ground observations, all stations | `stn_num`, `year`, `month`, `day`, `tmp_air_dry`, `tmp_air_wet`, `tmp_dew_pnt`, `hmd_rlt`, `prs_*`, `wind_*`, `cld_*` |
| `new_stn_table_web.csv` | station table | `stn_num`, `stn_name` (+ coordinates/height) |
| `new_isr_daily_data_web.csv` | daily precise max/min | `stn_num`, `time_obs`, `tmp_air_max`, `tmp_air_min` |

Period used by the paper: **2005-01-01 .. 2025-06-30** (the notebooks keep years >= 2005).

## Where the files go

```
data/raw_ims/
  new_isr_gnd_obs_web.csv
  new_stn_table_web.csv
  new_isr_daily_data_web.csv
```

## What consumes them

`preprocessing/01_Preprocessing_Hourly_Data.ipynb` (edit its path variables to point
here) turns the raw export into **one daily CSV per station**:

```
<DATA_ROOT>/Deep Learning Preprocess/Stations Daily Data 03_08_2025/<STATION_NAME>.csv
   columns: Date, tmp_air_dry, tmp_air_wet, tmp_dew_pnt, hmd_rlt, ...,
            max_dry_temp, min_dry_temp, max_wet_temp, min_wet_temp,
            max_heat_stress, min_heat_stress   (heat stress computed in the notebook)
```

Every downstream step (station-vector datasets, frames, LOSO study, baselines) reads
those per-station daily CSVs — see the Pipeline map in the top-level README.
