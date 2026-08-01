# Dataset summary (for the reproducibility appendix)

**Source:** Israel Meteorological Service daily station observations, 68 stations (union of 7 climate clusters), 2005-2025.

**Spatial frame:** study-wide 44x137 elevation-aware grid (SRTM DEM), shared across regions; only the target series is region-specific.

**Input window:** 180 days. **Forecast window:** 30 days. **Prediction days:** day-of-month in {1,7,14,21,28}.

**Targets:** order statistics p1/p3/p7/p15 = 1st/3rd/7th/15th hottest (coldest) day of the 30-day window.


## Feature channels (R^2 selection)

| feature | R^2 | k (elev/dist ratio) | used |
|---|---|---|---|
| prs_stn | 0.999 | 237.61 | no (dropped) |
| max_wet_temp | 0.907 | 14.80 | yes |
| tmp_air_wet | 0.872 | 20.93 | yes |
| min_wet_temp | 0.686 | 17.09 | yes |
| tmp_air_dry | 0.671 | 3.73 | yes |
| tmp_dew_pnt | 0.565 | 3.32 | yes |
| max_heat_stress | 0.535 | 2.67 | yes |
| min_heat_stress | 0.484 | 16.32 | yes |
| max_dry_temp | 0.484 | 2.32 | yes |
| min_dry_temp | 0.439 | 10.64 | yes |

-> 9 channels used: max_wet_temp, tmp_air_wet, min_wet_temp, tmp_air_dry, tmp_dew_pnt, max_heat_stress, min_heat_stress, max_dry_temp, min_dry_temp (prs_stn dropped).


## Chronological split (no leakage), #samples after DOM filter

| region | train | val | test | n_ext(test, p90) | train period | test period |
|---|---|---|---|---|---|---|
| Center | 813 | 143 | 238 | 31 | 2005-07-01..2019-01-28 | 2021-06-21..2025-06-01 |
| Negev | 814 | 143 | 239 | 46 | 2005-07-01..2019-01-21 | 2021-06-14..2025-06-01 |
| Northwest | 814 | 143 | 239 | 18 | 2005-07-01..2019-01-21 | 2021-06-14..2025-06-01 |
