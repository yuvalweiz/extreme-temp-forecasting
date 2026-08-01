# DATA.md — data inventory (raw → frames → datasets)

All under `/home/weizyuv/` (quote paths with the trailing-space `article ` dir). Datasets are NOT
in the repo (git-ignored / separate release); paths live in `src/pipeline/config.py`.

## 1. Raw source — IMS weather stations
- Per-station daily CSVs: `Deep Learning Preprocess/Stations Daily Data 03_08_2025/*.csv`
  (one per station, e.g. `GILAT_EXP_STATION.csv`; `Date` + daily-aggregated feature columns).
  Span ~2002–2025 (kept from 2005).
- Station metadata: `Deep Learning Preprocess/stations_meta_data.csv`; and
  `project codes/hybrid_frames_all_israel/grid_metadata.npz` = **68 stations** (station_names,
  station_lat/lon/elev) + the grid (lat_grid, lon_grid, elev_grid, pixel_matrix).
- Refreshed pull (optional, parked): `JIFX_2026/data/raw/` (8.21M hourly rows, 1920–2026).

## 2. Features (9) — the input channels
`k_values.csv` features minus `prs_stn` → `max_wet_temp, tmp_air_wet, min_wet_temp, tmp_air_dry,
tmp_dew_pnt, max_heat_stress, min_heat_stress, max_dry_temp, min_dry_temp`. All °C-scale
temperature/heat-stress; **pressure excluded**. Elevation is NOT a published input channel (the
9-ch datasets); a 10-ch (+elevation) `hybrid_pair_v1` lineage exists but was not used for results.

## 3. Frames — elevation-aware station→grid interpolation (all-Israel, shared)
- `Deep Learning Preprocess/Cluster All_ISRAEL/Daily Aggregation EXP_V2_CANONICAL_ALL_ISRAEL_HIGHRES/`
  `exponential_<feature>_<YYYY-MM-DD>.npy`, each **44×137**, 2005-01→2025-06.
- Three frame variants on disk: `EXP …`, `EXP_V2_CANONICAL …` (used by the published dataset),
  `IDW …` (plain distance — available for the EXP-vs-IDW ablation, no regeneration needed).
- Kernel: `w=exp(-k_exp·d_eff)`, `d_eff=√(d_km² + (0.001·k_f·Δh)²)`, k_exp=0.2. The elevation
  term is near-inert (weight L1 change 0.0035) → EXP ≈ distance interpolation. See ABLATIONS.md §D.

## 4. Target = cluster mean (per region); frames = all-Israel
- A region's **cluster** = a set of target stations (e.g. Center = 12 stations). Daily target =
  MEAN over the cluster's stations of daily max (hot) / min (cold), with a ≥30%-stations-present
  rule. The interpolated frames cover ALL 68 stations; the model predicts the cluster mean.

## 5. Model datasets (per region, per season) — `dataset_..._DOM_1_7_14_21_28`
Sample = `sample_<tag>.npz`: `X=(180,9,44,137)` fp16 frames, `y=(4,)` °C order stats, `pred_point`,
`input_days`, `out_days`, `features`. Window = `date_range(end=tag−1d, periods=180)`; tag day-of-
month ∈ {1,7,14,21,28}; target window = next 30 days.
- **HOT**: `Deep Learning Models/Cluster {Center,Negev,Northwest}/dataset_FULL_h180_next30_DOM_1_7_14_21_28`
  y=[1,3,7,15] hottest (desc daily max). norm `norm_stats_extremes_full_yNONE_v1.npz` (y_transform=none).
  Splits `split_{train,val,test}_yNONE_v1.csv`. Center 813/143/238; Negev/NW ~814/143/239. Test 2021–2025.
- **COLD/MIN**: `Cluster {…}/Winter Models/dataset_FULL_MIN_h180_next30_DOM_1_7_14_21_28`
  y=[1,3,7,15] coldest (asc daily min). norm `norm_stats_extremes_full_MIN.npz` (y_transform=asinh,
  has cold_p10/p05). Splits generic `split_{train,val,test}.csv`. Same frame structure as hot.
- Normalization: X channel-wise mean/std + y median/IQR, computed **TRAIN-ONLY** (no leakage);
  chronological splits.

## 6. Fair tabular ablation dataset (built by us) — `repo/data/stationvec_Center/`
`sample_<tag>.npz` with `X=(180,612)` [180 daily × 68 stations × 9 feats, station-major flat],
`y=(4,)` reused EXACTLY from the hot frame dataset. Same tags/splits/targets, differs only in
no-grid. norm `norm_stats.npz` (612-dim, train-only). 1194 samples; 1.86% test cells imputed
(ffill→train-mean). Symlinked space-free at `~/stationvec_Center`. See ABLATIONS.md §E.

## 7. Small runtime assets (committed in repo)
`src/pipeline/geo_channels.npy` (z-scored elev/lat/lon, 3×44×137, for GEO=1),
`lapse_correction.npy` (44×137 lapse map), `lapse_norm.npz`. Everything else (frames, datasets,
checkpoints) is git-ignored.

## 8. Published baselines / predictions (for eval + comparison)
`Models Evaluations/Predictions VS Actuals/{Summer,Winter}/<Region>/preds_*.csv`
(`preds_hot_*` = hottest anchor, `preds_soft_*` = 3/7/15, `preds_cold_*`, plus
`preds_{sarimax,prophet,tab_lstm}_*`). ~238–239 test rows/region. Significance harness data in
`Models Evaluations/full_significance_table.csv`.
