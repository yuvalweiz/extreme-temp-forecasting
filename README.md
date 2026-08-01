# Extreme-Temperature Forecasting for Electricity Peak-Demand Planning

Code, data recipes, and reproduction scripts for the paper's complete pipeline:
frame generation, training, baselines, and the evaluation that regenerates every
reported table.

**What the paper does (3 sentences).** Daily Israel Meteorological Service (IMS)
station observations are interpolated onto a 44x137 elevation-aware grid, giving one
multi-channel "frame" per day (9 weather channels), and a 180-day window of frames is
encoded frame-by-frame by a ConvNeXt-Tiny backbone whose per-day embeddings are fused
by a lightweight temporal-fusion head. The model is trained with an extreme-weighted
loss that extra-penalizes *under*-prediction on the hottest samples and predicts four
order statistics of the next 30 days — the 1st/3rd/7th/15th hottest day (coldest, in
the winter setting) — for three Israeli climate regions (Center, Negev, Northwest).
Against classical (SARIMAX, Prophet), per-station tabular deep (Tab-LSTM), and
zero-shot foundation-model (TimesFM 2.5, Moirai) baselines under an identical
seed-ensembled protocol, the spatial model is the most accurate on the month-ahead
extreme-day targets — significantly so against every baseline on the primary targets
pooled across regions, and against the matched spatial-structure ablation on the
coldest-day target in every region — with by far the lowest under-prediction of the
extremes, which is what electricity peak-demand planning needs.

---

## 1. Environment

```bash
conda create -n exttemp python=3.11 -y && conda activate exttemp
pip install -r requirements.txt          # torch 2.5.1, timm 1.0.24, ... (pinned)
```

* Evaluation runs on **CPU** (seconds to minutes). Training and the foundation-model
  baselines want a CUDA GPU (all paper runs: one RTX 4090, CUDA 12.1).
* The optional TimesFM / Moirai baselines need the extra pinned packages commented at
  the bottom of `requirements.txt`.
* Every command below is run **from the repo root**, with:

```bash
export PYTHONPATH="$PWD/src"
```

## 2. Data

Two environment variables control all data resolution (`src/repo_paths.py` is the
single source of truth; every script resolves paths through it):

| variable | meaning | default |
|---|---|---|
| `REPO_DATA` | bundled/release data directory | `<repo>/data` |
| `DATA_ROOT` | root of the large optional data (author layout) | `$HOME` |

### 2.1 Bundled with the repo (git)

Small files committed to git — enough to **regenerate every reported table** and to
verify the code paths:

* `data/published_predictions/{Summer,Winter}/` — the published per-model prediction
  CSVs (ours + all baselines, per region, both seasons) that the paper's tables are
  computed from.
* `data/dataset_meta/<Region>[_MIN]/` — the canonical chronological split CSVs,
  normalization stats, and `y_values_*.csv` (the per-sample targets, extracted
  verbatim from the dataset, so thresholds/tables recompute from a bare clone).
* `data/grid_metadata.npz`, `data/k_values.csv`, `data/stations_meta_data.csv`,
  `data/features_order.txt` — grid + station metadata.
* `results/timesfm_*.csv`, `results/moirai_*.csv` — foundation-model baseline outputs.
* `src/pipeline/{geo_channels.npy, lapse_correction.npy, lapse_norm.npz}` — small
  precomputed model assets (`GEO=1` / `LAPSE=1` work out of the box).

### 2.2 Data release (unpack into `<repo>/data/`)

The **station-vector datasets** (`data/stationvec_<Region>/`,
`data/stationvec_MIN_<Region>/`; ~400 MB each) and the **interpolation weight
matrices** (`data/interp_weights/`; ~260 MB) are distributed as a data release
(too big for git). Unpack them into `<repo>/data/` (or point `REPO_DATA` at them).
They are enough to **train the full spatial model** (frames are synthesized on the
fly as `weights x station-vector`, the validated equivalent of the frame bank) and
to run the fair-tabular ablation. Both are also **fully regenerable** from the raw
data with the scripts below.

### 2.3 Obtain the raw IMS data (manual, free)

Raw observations come from the IMS open-data portal — https://ims.data.gov.il
(portal of https://ims.gov.il/en). Download the archive files
`new_isr_gnd_obs_web.csv`, `new_stn_table_web.csv`, `new_isr_daily_data_web.csv`
into `data/raw_ims/` (**exact expected format: `data/raw_ims/README.md`**), then run
`preprocessing/01_Preprocessing_Hourly_Data.ipynb` to produce the per-station daily
CSVs:

```
<DATA_ROOT>/Deep Learning Preprocess/Stations Daily Data 03_08_2025/<STATION_NAME>.csv
```

Those daily CSVs feed everything downstream (datasets, frames, LOSO, baselines).
The full-fidelity extras used by some author-side scripts live under the same
`DATA_ROOT` (see `src/repo_paths.py` docstring for the exact layout): the ~10 GB/region
canonical frame datasets, the published checkpoints (`verify_weights.py`), the on-disk
frame banks, and SRTM `.bil` DEM tiles for `rebuild_frames.py`.

## 3. Pipeline map (raw data -> tables)

```
raw IMS export                      data/raw_ims/                        [manual download]
  |   preprocessing/01_Preprocessing_Hourly_Data.ipynb                   [notebook]
  v
per-station daily CSVs              <DATA_ROOT>/.../Stations Daily Data 03_08_2025/
  |   src/data/build_stationvec.py                                       [script, validated]
  |------------------> station-vector datasets   data/stationvec_*       (also in release)
  |   src/pipeline/compute_interp_weights.py                             [script, ~1 min CPU]
  |------------------> station->grid weights     data/interp_weights/    (also in release)
  |   preprocessing/02_Frames_and_Grid_Preprocess_with_height_V2.ipynb   [notebook]
  |   (script equivalent: src/data/rebuild_frames.py)
  \------------------> daily frame banks         (optional; training does NOT need them)

train    python -m pipeline.train   DATASET_DIR=<stationvec> PAPERW=<weights .npz>
  \------------------> runs/<run>/preds_{val,test}.csv + meta.json + best.pt

evaluate src/eval/reproduce_metrics.py + run_significance.py + per_target_metrics.py
  \------------------> results/*.csv  ==  the paper's tables (to 0.0000)
```

Interpolation-quality study: `src/eval/loso_interpolation.py` (daily CSVs -> LOSO MAE
per kernel). Baselines: `src/baselines/` (daily CSVs -> cluster series -> targets).

### 3b. Learned station-to-grid interpolation (NN variant)

The learned interpolation replaces the fixed elevation-aware kernel with a light
attention set-regressor (~200k parameters, one network for all 9 features; weights and
all normalization statistics fitted **only on training-period days**, so the time split
is preserved by construction). The fitted network is versioned in this repo
(`experiments_real/nninterp/nninterp.pt`, 512 KB) for exact reproduction; the three
scripts regenerate everything from the daily CSVs:

```bash
cd experiments_real/nninterp
python build_daily_sv.py       # daily station matrix + train-period norm stats (~1 min)
python train_nninterp.py       # trains the interpolation network (~10 min GPU)
                               #   + prints the LOSO test-period MAE (paper: 1.04 C
                               #     vs 1.18 C for the tuned kernel)
python gen_nn_bank.py          # writes the full daily frame bank framebank_nn/ (~15 min)
```

Training the forecaster on the learned frames = the standard training command plus:

```bash
FRAMEBANK=<...>/framebank_nn FRAMEBANK_NS=<...>/framebank_nn/norm_stats_x.npz \
CACHE=0 WINDOW=180 python -m pipeline.train ...   # rest identical
```

## 4. Quickstart

### 4.1 Reproduce the paper's tables (CPU; repo + git-bundled data only)

```bash
export PYTHONPATH="$PWD/src"
python src/eval/reproduce_metrics.py    # metric table  -> results/summer_metrics_*.csv   (PASS @ max|d|=0.0000)
python src/eval/per_target_metrics.py   # per-target p1/p3/p7/p15 MAE -> results/summer_per_target.csv
python src/eval/run_significance.py     # moving-block bootstrap + Holm -> results/summer_significance.csv
                                        #   (B=10000, ~10 min CPU; N_BOOT=1000 for a quick pass)
# full-evaluation-set protocol (the resubmission's tables): row-level actual-vs-predicted
# CSVs + per-region MAE/RMSE for BOTH seasons, verified vs the frozen references
python src/eval/export_actual_vs_predicted.py   # -> results/actual_vs_predicted/ (PASS @ max|d|=0.000000)
python src/eval/run_significance_unified.py     # -> results/actual_vs_predicted/significance_unified.csv
# FINAL manuscript numbers: results/actual_vs_predicted_corrected/ bundles the row-level
# predictions of the corrected seed-ensembled models (see its README); every table cell
# of the final paper recomputes from those files.
python src/eval/dataset_stats.py        # dataset summary table -> results/dataset_stats.md
python src/eval/downstream_proxy.py     # peak-demand planning proxy -> results/downstream_proxy.csv
python src/eval/plot_pred_vs_true.py    # p1 actual-vs-predicted figures -> results/figures/
```

The extreme threshold is recomputed from TRAIN y-values on the fly, resolving the y
source in this order: canonical frame dataset (`DATA_ROOT`) -> station-vector dataset
(data release) -> git-bundled `data/dataset_meta/<Region>/y_values_*.csv` (verbatim
targets). All three give identical thresholds (verified) — a bare `git clone` +
`pip install` regenerates the tables with no downloads.

### 4.2 Train the model (GPU; repo + data release)

The published configuration per region is loaded automatically (`PUBLISHED` in
`src/pipeline/config.py`: Center a2/b1, Negev a3/b1.5, Northwest a1/b0). Frames are
synthesized on the fly from the bundled weights + station vectors:

```bash
export PYTHONPATH="$PWD/src"
# smoke test (2 epochs) — the full path: synth frames -> ConvNeXt -> TFT -> preds CSVs
REGION=Center RUN_TAG=smoke EPOCHS=2 SEED=111 WARMUP=2 DETERMINISTIC=1 CACHE=0 \
  DATASET_DIR="$PWD/data/stationvec_Center" \
  PAPERW="$PWD/data/interp_weights/paper_opt.npz" \
  CKPT_ROOT="$PWD/runs/Center" \
  python -m pipeline.train

# real run: drop EPOCHS (default 120, patience 30) and use 3 seeds {111,222,333}
# fair tabular ablation (station-vector -> temporal head, no CNN):
REGION=Center RUN_TAG=tab STATIONVEC=1 DATASET_DIR="$PWD/data/stationvec_Center" \
  CKPT_ROOT="$PWD/runs/Center" python -m pipeline.train
# cold/winter task:
REGION=Center COLD=1 K=1 RUN_TAG=cold DATASET_DIR="$PWD/data/stationvec_MIN_Center" \
  PAPERW="$PWD/data/interp_weights/paper_opt.npz" CKPT_ROOT="$PWD/runs/Center_cold" \
  python -m pipeline.train
```

Each run writes `CKPT_ROOT/<RUN_TAG>__[COLD__]<head>__a<ALPHA>__b<BETA>__s<SEED>/`
with `preds_val.csv`, `preds_test.csv` (+ `preds_*_topk.csv` when `SAVE_TOPK>0`) and
`meta.json` (full config + metrics). Evaluation consumes these CSVs directly.

To train on the **canonical disk-frame datasets** instead (full release), just set
`DATA_ROOT=/path/to/full_release` and drop `DATASET_DIR`/`PAPERW` — byte-faithful to
the published runs.

**Env vars read by `pipeline.train`** (defaults = the published configuration):

| var | meaning |
|---|---|
| `REGION` | `Center` \| `Negev` \| `Northwest` (dataset + published anchor config) |
| `SEED` `EPOCHS` `PATIENCE` `DROPOUT` `RUN_TAG` `CKPT_ROOT` | standard knobs (120/30/0.1) |
| `ALPHA` `BETA` | extreme under-multiplier / under-hinge weight (per-region default) |
| `DATASET_DIR` | dataset dir override (canonical frames, stationvec, or IDW dataset) |
| `PAPERW` | interp-weights `.npz` -> synthesize frames on the fly (+`PAPERW_MASKS/COV/ANOM`) |
| `STATIONVEC` | `1` = fair 612-dim tabular ablation (no CNN) |
| `TABULAR` | `1` = region-mean 9-dim tabular control (canonical frame dataset only) |
| `COLD` + `K` | winter/coldest task (geometric out-weights, p10 tail, percentile-k loss) |
| `WINDOW` `TARGETS` `EXTREME_Q` `OUT_WEIGHTS` | history / order-stat ranks / tail q / loss weights |
| `HEAD` | `temporalfusion` (published) \| `lstm` \| `patchtst` \| `itransformer` |
| `BACKBONE` `POOL` `SPATIAL_STAGE` `D_MODEL` `NHEAD` `HEAD_LAYERS` | architecture ablations |
| `GEO` `LAPSE` | append static elev/lat/lon channels / lapse-rate-corrected frames |
| `SCHEDULER` `EMA` `NO_WD_NORM` `WARMUP` | optimization ablations (all off = published) |
| `SELECT` `SELECT_Q` `SAVE_TOPK` | validation-selection variants (+ mean-of-top-k preds) |
| `PRETRAINED` | path to an in-domain-pretrained encoder (`pipeline/pretrain_frames.py`) |
| `DETERMINISTIC` | `1` = cudnn deterministic |
| `CACHE` | `1` = RAM-cache samples (default; use `CACHE=0` on small-RAM machines) |
| `HYBRID` `D_SV` `MULTITASK` `DENSE_PRETRAIN` `BACKBONE3D` `TRAINVAL` `FIXED_STOP` `DIAG_TEST` | documented extensions (see `train.py` docstring) |

### 4.3 LOSO interpolation study (CPU; needs the per-station daily CSVs)

Leave-one-station-out fidelity of every interpolation kernel — the paper's
elevation-aware-interpolation evidence:

```bash
export PYTHONPATH="$PWD/src"
DATA_ROOT=/path/to/data python src/eval/loso_interpolation.py            # all 9 features
# quick pass (one feature): add LOSO_FEATURES=max_dry_temp
# leakage-free kernel-selection variant (train period only): LOSO_END=2019-01-28
```

### 4.4 Regenerate datasets + weights (the "frame generation" steps)

```bash
export PYTHONPATH="$PWD/src"
# station->grid interpolation weight matrices (bundled inputs; ~1 min CPU, self-validating)
python src/pipeline/compute_interp_weights.py
# station-vector datasets from the daily CSVs (validated vs the bundled release)
DATA_ROOT=/path/to/data REGION=Center \
  OUT_DIR=/tmp/sv_Center VALIDATE="$PWD/data/stationvec_Center" \
  python src/data/build_stationvec.py
# frames / grid: preprocessing/02_...V2.ipynb; script check: src/data/rebuild_frames.py (full release)
```

### 4.5 Baselines & checkpoint verification (optional; full release / extra deps)

```bash
export PYTHONPATH="$PWD/src"
DATA_ROOT=/path/to/data REGION=Center python src/baselines/run_timesfm.py     # TimesFM 2.5 zero-shot
DATA_ROOT=/path/to/data REGION=Center python src/baselines/run_moirai.py      # Moirai-1.1-R zero-shot
DATA_ROOT=/path/to/data REGION=Center HEAD=temporalfusion python src/baselines/train_tabular.py
DATA_ROOT=/path/to/data python src/eval/verify_weights.py   # published ckpts -> saved preds (max|d|~0)
```

All baselines reduce each 30-day forecast to p1/p3/p7/p15 with the identical
sort-descending `[0,2,6,14]` protocol (`docs/BASELINE_PROTOCOL.md`), so the same
`src/eval` code scores every model.

## 5. Repository layout

```
README.md  LICENSE  requirements.txt
preprocessing/                # raw IMS -> daily CSVs -> grid/frames (original notebooks + README)
src/
  repo_paths.py               # central path resolution (REPO_DATA / DATA_ROOT)
  pipeline/                   # CANONICAL training pipeline (validated port of run_grid.py)
    train.py                  #   entry point: python -m pipeline.train (env-driven)
    config.py  data.py  model.py  loss.py  metrics.py
    compute_interp_weights.py #   station->grid weight matrices (PAPERW inputs)
    compute_kriging_weights.py  build_station_masks.py  stack_weights.py
    pretrain_frames.py  compute_lapse.py  build_idw_mmap.py  precompute_cache.py
    geo_channels.npy  lapse_correction.npy  lapse_norm.npz
  eval/                       # everything that regenerates the paper's numbers
    eval_lib.py  reproduce_metrics.py  run_significance.py  per_target_metrics.py
    dataset_stats.py  downstream_proxy.py  plot_pred_vs_true.py
    loso_interpolation.py  verify_weights.py  ensemble_eval.py
  baselines/                  # series.py (targets harness), run_timesfm.py, run_moirai.py, train_tabular.py
  data/                       # interpolation kernels + dataset/frame builders
    interpolation.py  build_stationvec.py  rebuild_frames.py  frame_window_dataset.py
data/                         # bundled small data (git) + data-release contents (see § 2)
results/                      # regenerated tables/figures + bundled baseline outputs + REGISTRY.md
docs/                         # protocol, hyperparameters, reviewer evidence, strategy
legacy/                       # superseded early reimplementation — provenance only, do not use
```

## 6. Validation status (what "reproduces" means here)

* `reproduce_metrics.py` matches the published summer metric table to **max|d| = 0.0000**
  (all models, hot + soft), from the bundled prediction CSVs + recomputed thresholds.
* `run_significance.py` rebuilds the significance table with the documented cross-region
  pooling correction (`docs/STRATEGY.md`).
* `series.py` (baseline harness) reproduces the paper's targets to **0.0000**.
* `build_stationvec.py` regenerates the bundled station-vector datasets from the daily
  CSVs to float32 precision (max|dX| ~ 7e-5, y exact).
* `compute_interp_weights.py` self-validates (row-stochasticity, elevation spot checks).
* `verify_weights.py` (full release) reloads the published checkpoints and matches the
  saved predictions (max|d| ~ 0).
* Selection protocol: configs/epochs chosen on **validation only**, test read once;
  chronological splits; extreme threshold from TRAIN only. Experiment log:
  `results/REGISTRY.md`.

## 7. Limitations (honest notes for replicators)

* **The raw IMS download is manual.** IMS offers the data freely but behind its own
  portal; we do not redistribute the raw export or the per-station daily CSVs. All
  *derived* artifacts needed to reproduce the paper's tables ship with the repo/release.
* **The preprocessing notebooks are provenance copies.** They document (and performed)
  raw->daily and frame generation, but contain the author's local paths — edit the path
  variables before re-executing. Their scripted equivalents
  (`build_stationvec.py`, `rebuild_frames.py`, `compute_interp_weights.py`) are
  path-parameterized and tested.
* **The ~10 GB/region canonical frame datasets and published checkpoints** are in the
  full data release, not git. Training reproduces without them via validated on-the-fly
  frame synthesis (`PAPERW`); `verify_weights.py` and `rebuild_frames.py --validate`
  need the full release.
* **Foundation-model baselines** download large pretrained weights from Hugging Face on
  first run; their outputs are bundled in `results/` so the tables do not depend on
  rerunning them.

## 8. License & citation

MIT — see `LICENSE` (author line intentionally anonymized during double-blind review).

```bibtex
@article{ANONYMIZED2026extreme,
  title   = {Extreme Temperature Forecasting for Electricity Peak Demand Planning},
  author  = {ANONYMIZED — filled on acceptance},
  journal = {Data Mining and Knowledge Discovery (under review)},
  year    = {2026},
  note    = {Code: this repository}
}
```
