# PIPELINE.md — end-to-end, raw data → trained model → evaluation

The published method as 4 stages. Originals are the source of truth; `src/pipeline/` is a validated
1:1 port (loads published state-dicts 0/0 mismatch, reproduces metrics to 0.0000). Same input frames
for hot and cold — **only the target (output) and loss differ**.

## Stage 1 — Raw stations → elevation-aware grid frames
Input: per-station daily CSVs (68 stations, 9 features). 
- `k_coef_extraction.py` → per-feature elevation coefficient `k_f` (`k_values.csv`).
- `generate_hybrid_frames.py` (+ playground preprocess notebook): EXP interpolation onto a 44×137
  all-Israel grid — `w=exp(-0.2·d_eff)`, `d_eff=√(d_km²+(0.001·k_f·Δh)²)` (elevation term inert →
  effectively distance interpolation). 
Output: `exponential_<feat>_<date>.npy` daily frames (`Daily Aggregation EXP_V2_CANONICAL…`).
Shared across all regions. (IDW variant also on disk for the interpolation ablation.)

## Stage 2 — Frames → windowed samples + order-stat targets
Playground notebook cells 9 (build) + 13 (normalize) ≡ `data_preprocess.py`/`build_hybrid_datasets.py`.
- For each anchor (DOM∈{1,7,14,21,28}): X = 180 daily frames ending at anchor (`date_range(end=
  anchor, periods=180)`); target = cluster-mean series over next 30 days, order stats at ranks
  [1,3,7,15] — **descending on daily max (HOT)** or **ascending on daily min (COLD)**.
- Chronological split (test 0.20, val 0.15 of trainval); X mean/std + y median/IQR **TRAIN-ONLY**.
Output: `dataset_FULL_h180_next30_DOM…` (HOT, yNONE) and `Winter Models/dataset_FULL_MIN_…` (COLD,
asinh). See DATA.md.

## Stage 3 — Training  (same model both seasons: ConvNeXt-Tiny → temporal head → OutputAffine)
Backbone: ConvNeXt-Tiny (ImageNet) per frame → (B,T,768); head: TemporalFusionLite (or LSTM);
d_model=256, nhead=4, layers=2. AdamW (bb 1e-5, head 1e-4), grad-clip 1, AMP, batch 4.
- **HOT** (`run_grid.py` ≡ `pipeline.train`): loss `WeightedMAE_ExtremeUnderOnly` (MAE; under-
  penalty α ONLY on extremes where true_hot≥train-p95) + `ExtremeUnderHinge` (β·ReLU(true−pred) on
  ext). **out_weights=(1,1,1,1) equal**, y_transform=none, wd 1e-4, **no scheduler/EMA**,
  selection = avg3(val_mae_all, val_mae_ext, under_rate_ext). Per-region loss: Center a2b1, Negev
  a3b1.5, NW a1b0.
- **COLD** (`run_grid_cold_winter.py` ≡ `pipeline.train COLD=1`): loss `PercentileWeightedMAE_Min`
  (ECDF-rank sample weights emphasizing colder-than-median + α on "too-warm" errors pred>true on
  cold) + optional hinge. **out_weights=(1,0.5,0.25,0.125) geometric**, y_transform=asinh, wd_bb
  1e-3, **cosine LR + EMA-0.35 val-score smoothing**, extreme = train-p10 (low tail), selection =
  avg(mae_cold_all, mae_cold_ext).
- **SOFT model** (secondary, `run_grid_avg3.py`): a SEPARATE model predicting only ranks 3/7/15
  (out_dim 3), percentile-weighted loss. Exists for both seasons.
- Known VARIANCE issue (both): the avg3 early-stop is noisy (best_ep swings 1–33 → ~1°C run-to-run).
  Stabilized in the clean pipeline via `DETERMINISTIC=1` + `WARMUP=15` min-epoch floor + 3-seed mean.

## Stage 4 — Evaluation
- Metrics per target: **MAE_all, MAE_ext (top/bottom 10% = p90/p10), Mean MAE=(all+ext)/2** (main
  ranking), + RMSE. Aggregated across the 3 regions for the headline tables. (`eval_lib.py`,
  notebook results cells 27–49.)
- Significance: moving-block bootstrap (B=10000, block 4 all / 2 ext) + Holm-Bonferroni, per region
  (`run_significance.py`; the pooled-ALL rows were once 3× inflated — fixed).
- Baselines: SARIMAX/Prophet/Tab-LSTM (author preds) + TimesFM/Moirai (`src/baselines`).

## Reproduce (commands)
```
PYTHONPATH="/home/weizyuv/article /repo/src"
cd src/eval && python reproduce_metrics.py        # paper table, max|Δ|=0.0000
python run_significance.py                          # bootstrap + Holm
# train hot (GPU):
REGION=Center DETERMINISTIC=1 WARMUP=15 CACHE=1 CKPT_ROOT=... python -m pipeline.train
# train cold (GPU):
COLD=1 REGION=Center DETERMINISTIC=1 WARMUP=15 CACHE=1 CKPT_ROOT=... python -m pipeline.train
# ablation = same + env (WINDOW / TARGETS / POOL / BACKBONE / TABULAR / STATIONVEC / LAPSE / …)
```

## Improvements we found (DEFERRED from the 1:1 article — see ARTICLE.md, DAMI_ARTICLE_GUIDE.md)
Recipe (V2 backbone + cosine/EMA + in-domain pretrain + [1,5,10,15] targets) beats published on the
HOT anchor per-region; must be re-expressed in Mean MAE @10% (3-region) for a headline claim. Cold-
anchor rollout of the recipe pending. Soft ≈ tied. Elevation levers (channel + lapse) don't help
single-cluster (constant per cluster). Fair tabular: spatial wins the extreme tail (ABLATIONS.md).
