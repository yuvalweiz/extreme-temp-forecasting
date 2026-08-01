# preprocessing/ — raw IMS export -> daily station CSVs -> grid + frames

Verbatim copies of the author's original preprocessing notebooks (kept as-is for
provenance; edit the path variables at the top of each notebook to your machine).

## Order

1. **`01_Preprocessing_Hourly_Data.ipynb`** — raw IMS export -> per-station daily CSVs.
   * Input: the three IMS portal files in `data/raw_ims/` (see `data/raw_ims/README.md`).
   * Steps: split the country-wide hourly CSV per station -> drop metadata columns ->
     compute heat stress -> aggregate to daily (mean + max/min) -> add a `Date` column ->
     merge the higher-precision daily `tmp_air_max`/`tmp_air_min`.
   * Output: `<DATA_ROOT>/Deep Learning Preprocess/Stations Daily Data 03_08_2025/<STATION>.csv`
     (68 stations are used downstream; `data/grid_metadata.npz` fixes the canonical
     station order).

2. **`02_Frames_and_Grid_Preprocess_with_height_V2.ipynb`** — daily CSVs -> 44x137 grid +
   daily interpolated frames (`02b_..._V1.ipynb` is the earlier version, kept for
   provenance).
   * Input: the per-station daily CSVs + SRTM DEM `.bil` tiles
     (`<DATA_ROOT>/Deep Learning Preprocess/height_data/bil files/`) + `data/k_values.csv`.
   * Output: per-day per-feature rasters
     `<DATA_ROOT>/Deep Learning Preprocess/Cluster All_ISRAEL/Daily Aggregation EXP .../
      exponential_<feature>_<YYYY-MM-DD>.npy` (and `idw_*` variants).
   * Script equivalents in the repo: `src/data/rebuild_frames.py` (grid + SRTM elevation +
     frame reconstruction, validates against a stored sample) and
     `src/pipeline/compute_interp_weights.py` (precomputes the station->grid weight
     matrices in `data/interp_weights/` so frames can be synthesized on the fly without
     a frame bank — this is what training's `PAPERW` mode uses).

After step 1 you can already run, with no notebook re-execution needed:

* `src/data/build_stationvec.py` — rebuilds the station-vector datasets
  (`data/stationvec_*`) from the daily CSVs; validated to reproduce the bundled
  datasets to float32 precision (`VALIDATE=...`).
* `src/eval/loso_interpolation.py` — the leave-one-station-out interpolation study.
* `src/baselines/*` — the non-spatial baselines (cluster series + order-stat targets).

The per-sample **frame** datasets (`dataset_FULL_h180_next30_DOM_1_7_14_21_28`, ~10 GB
per region) used by the published runs were assembled in the author's Stage-3 modeling
notebook; the repo trains WITHOUT them by synthesizing frames on the fly
(`PAPERW` = interpolation weights x station vectors — see README § Training). The
in-repo window-assembly equivalent for on-disk frame banks is
`src/data/frame_window_dataset.py` / `src/pipeline/build_idw_mmap.py`.
