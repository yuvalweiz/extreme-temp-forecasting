# CODE.md — codebase map

Repo `/home/weizyuv/article /repo/` (deploy-ready). Env: conda `timesfm311` (torch 2.4.1+cu121,
timm 1.0.15); python `/home/weizyuv/.conda/envs/timesfm311/bin/python`. Run with
`PYTHONPATH="/home/weizyuv/article /repo/src"`.

## Canonical pipeline — `src/pipeline/` (faithful port of the originals, validated 0/0 keys)
- `config.py` — `Config`, per-region `REGION_DATASET` (hot) + `REGION_DATASET_MIN` (cold),
  `PUBLISHED` per-region loss (Center a2b1, Negev a3b1.5, NW a1b0), `COLD_OUT_WEIGHTS` (geometric),
  `COLD_EMA_ALPHA_SCORE`.
- `data.py` — `FullSampleDataset` (npz frames + norm + geo/tabular/lapse), `MmapDataset`,
  `StationVecDataset` + `build_stationvec_data` (fair tabular), `build_data`.
- `model.py` — `ConvNeXtTiny_WithHead` (backbone + head + OutputAffine); heads TFT/LSTM/PatchTST/
  iTransformer; `ConvNeXtTinyBackbone` (pool avg/max/avgmax/attn; `spatial_stage` = A1 token
  preservation); `HeadOnlyModel` (non-spatial control).
- `loss.py` — HOT: `WeightedMAE_ExtremeUnderOnly_Celsius` + `ExtremeUnderHinge`. COLD:
  `PercentileWeightedMAE_Min` + `PercentileWeightedHinge_Min`.
- `metrics.py` — hot: `collect_pred_table`, `val_select_score_avg3`, `suite`; cold:
  `compute_cold_metrics`, `val_select_score_cold`, `suite_cold`.
- `train.py` — `run_one_config` (hot) + `run_one_config_cold` (COLD=1) + `build_cold_data`; env-
  driven `main()`. Entry: `python -m pipeline.train`.
- `pretrain_frames.py` — in-domain masked-AE pretraining (leak-free frames ≤ train cutoff) →
  encoder loaded via `PRETRAINED`.
- `compute_lapse.py` — builds the lapse-rate correction map + lapse norm (A3, memory-wise).
- assets: `geo_channels.npy`, `lapse_correction.npy`, `lapse_norm.npz` (committed).

## Env knobs (train.py) — one command, many ablations
`REGION, HEAD, ALPHA, BETA, DROPOUT, SEED, EXTREME_Q, WINDOW, OUT_WEIGHTS, TARGETS, POOL{avg,max,
avgmax,attn}, BACKBONE{convnext_tiny,convnextv2_tiny,convnextv2_nano}, D_MODEL, NHEAD, HEAD_LAYERS,
NO_WD_NORM, SCHEDULER{cosine}, EMA, PRETRAINED, GEO, TABULAR, SPATIAL_STAGE, LAPSE, STATIONVEC,
COLD, K, DATASET_DIR, CKPT_ROOT, CACHE, DETERMINISTIC, WARMUP, EPOCHS, PATIENCE, RUN_TAG`.
Stabilized protocol = `DETERMINISTIC=1 WARMUP=15 CACHE=1` + 3 seeds.

## Evaluation — `src/eval/`
`eval_lib.py` (metrics + moving-block bootstrap + Holm; reproduces paper to 0.0000),
`reproduce_metrics.py`, `run_significance.py`, `per_target_metrics.py` (p1/p3/p7/p15),
`dataset_stats.py`, `verify_weights.py` (load published ckpts, match saved preds),
`plot_pred_vs_true.py`, `ensemble_eval.py`, `downstream_proxy.py`.

## Baselines — `src/baselines/`
`series.py` (target harness, reproduces targets 0.0000), `run_timesfm.py`, `run_moirai.py`
(foundational, zero-shot), `train_tabular.py` (Tab-LSTM).

## Interpolation / frames — `src/data/`
`interpolation.py` (`weights_paper`, γ-tunable, elevation on/off — the A3 paper-accurate kernel),
`rebuild_frames.py` (grid + SRTM elevation + frame regen), `frame_window_dataset.py`.

## Legacy — `legacy/` (NOT used for any reported result)
`train_reimpl/` (`train_grid_hot.py`, `train_from_frames.py`, `models.py`) — early reimplementation
that produced INVALID early experiments; `aggregate_experiments.py`. Kept for provenance only.

## Originals (OUTSIDE the repo — NEVER modified)
`~/run_grid.py` (hot, ≡ pipeline hot), `~/run_grid_cold_winter.py` (cold, ≡ pipeline COLD=1),
`~/run_grid_{2,3,4,avg3,cold_avg3,reseed}.py` (variants; avg3 = the SOFT model),
`~/project codes/{generate_hybrid_frames,build_hybrid_datasets,data_preprocess,k_coef_extraction}.py`,
playground notebook `article /originals/Modeling And Predictions Summer (STAGE 3) …PreTrained.ipynb`
(dataset construction in early cells + results cells). See PIPELINE.md for how these compose.

## Reproduce / train / cold (quick)
- Reproduce paper: `cd src/eval && python reproduce_metrics.py` (→ 0.0000), `run_significance.py`.
- Train hot: `PYTHONPATH=src REGION=Center DETERMINISTIC=1 WARMUP=15 CACHE=1 CKPT_ROOT=... python -m pipeline.train`.
- Train cold: same + `COLD=1` (auto: MIN dataset, geometric weights, p10, cosine+EMA).
- Docs: [[ARTICLE.md]], [[DATA.md]], [[PIPELINE.md]], ABLATIONS.md, DAMI_ARTICLE_GUIDE.md.
