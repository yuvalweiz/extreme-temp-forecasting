# Baseline & target protocol (answers R3: "baselines not described algorithmically")

Every model — spatial and non-spatial — predicts the **same four order
statistics** on the **same splits**, so comparisons are apples-to-apples. This
is the protocol extracted from `Basline Methods.ipynb` (cell 1–3).

## Target definition (shared by all models)
- A **region** is a fixed list of cluster stations (e.g. Center = 12 stations:
  ASHDOD PORT, ASHQELON PORT, BEIT JIMAL, BET DAGAN, DOROT, GAT, HAFEZ HAYYIM,
  NAHSHON, NEGBA, NIZZAN, QEVUZAT YAVNE, TEL AVIV COAST).
- Daily cluster series: per day, aggregate the target across cluster stations —
  `max_dry_temp` → **max** (hottest), rain → sum, others → mean. (Cold/winter
  uses `min_dry_temp` → min.)
- For a prediction made at day `t` (only days-of-month {1,7,14,21,28} are used),
  look at the **next `FORECAST_DAYS = 30` days**, sort that 30-value daily series
  descending, and read off the order statistics:

  ```
  p1  = sorted[0]    # hottest day in the window
  p3  = sorted[2]    # 3rd hottest
  p7  = sorted[6]    # 7th hottest
  p15 = sorted[14]   # 15th hottest  (~median of the 30-day window)
  ```

- Input history = `HISTORY_DAYS = 180` days ending at `t`.
- Split: `test_frac = 0.20`, `val_frac_of_train = 0.15`, **chronological**
  (no shuffling), so val/test are strictly later than train → no leakage.

**Why p1/p3/p7/p15** (answers R2/R3): p1 is the peak-demand-critical maximum;
p15 (~15th of 30) marks the boundary of the upper tail (roughly the monthly
median of hot days); p3/p7 sample the tail in between. They are a compact,
roughly geometric summary of the upper tail rather than a full trajectory.
(Ablation A1 will test alternative sets, e.g. {1,5,10,20}.)

## How each non-spatial baseline produces the order statistics
- **SARIMAX / Prophet (forecast→reduce):** fit on the training daily cluster
  series; produce a 30-day-ahead daily forecast for the window after `t`; apply
  the exact `sort-descending → [0,2,6,14]` reduction above to the forecast.
- **Tab-LSTM (direct):** an LSTM over the 180-day multivariate cluster history
  (Israel-wide tabular features) predicts the 4 order statistics directly. This
  is the **no-spatial control**: same targets, same loss family, no frames.
- **(P2 additions) TimesFM / Chronos:** zero-shot / light fine-tune on the daily
  cluster series → 30-day forecast → same reduction. **Tabular PatchTST /
  iTransformer:** direct order-statistic regression like Tab-LSTM.

## Spatial models (the proposed method)
Identical targets/splits, but the input is the **nationwide station→grid frame
stack** (180×9×44×137) encoded by ConvNeXt-Tiny + temporal head (TFT/LSTM),
trained with the percentile-weighted extreme loss. The contrast spatial-vs-
Tab-LSTM isolates the value of spatial information.

## Metrics (per `eval_lib.py`, verified to reproduce the paper 0.0000)
- **hot/cold target:** MAE/RMSE on p1.
- **soft target:** MAE/RMSE on `mean(p3,p7,p15)` (abs-of-mean; summer convention).
- Reported **all-case** and **extreme-case** (true p1 ≥ train p90 / ≤ train p10),
  always shown separately so the aggregate is never the only number (answers
  R3/R4 "averaging is controversial").
- Significance: moving-block bootstrap (B=10000, block 4 all / 2 ext) + Holm.
