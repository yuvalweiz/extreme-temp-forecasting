# Experiment Registry — single source of truth

Mission: make the extreme-temperature spatiotemporal method SOTA and answer the
ECML-PKDD reviews. Beat SARIMAX/Prophet/Tab-LSTM AND foundational models, with
clean ablations and full reproducibility. Constraints: 3 clusters, 1 country.
Core method fixed: elevation-aware station→grid frames → ConvNeXt-Tiny → temporal
head (TFT/LSTM) → percentile-weighted extreme loss → p1/p3/p7/p15 targets.

**ADOPTION RULE (author-set, strict):** A candidate replaces the published
ConvNeXt-TFT ONLY IF it improves it on **ALL FOUR**: val-all, val-extreme,
test-all, test-extreme (seed-averaged). Better-on-val-but-not-test = noise →
rejected, keep incumbent. As of latest run, PatchTST & LSTM rejected (win val-all
only); incumbent TFT (a3b0 seed-avg) = val 2.571/2.152, test 2.066/2.316 stands.

**SELECTION & FAIRNESS PROTOCOL (inviolable):**
- Config/hyperparameters chosen by **validation only** (EMA of avg(val_all,val_ext);
  train-only p90 extreme threshold; chronological split). **Test is read once** for the
  val-selected config. We NEVER scan configs and keep the best *test* (that is leakage).
- "best-by-test" is shown in aggregators only as a flagged DIAGNOSTIC, never reported.
- **Fairness:** identical protocol for OURS and the COMPETITION. Headline = single
  val-selected model vs single val-selected Tab-LSTM/baselines (apples-to-apples).
  Seed-averaging/ensembling, if used, is a SEPARATE row applied symmetrically (e.g.
  ensemble-ours vs ensemble-TabLSTM) — never ours-boosted vs their-single.
- To make val→test agree legitimately: reduce variance (seed-avg, ensemble) + improve
  generalization (regularization) so the val-selected model is robustly strong on test.
- The paper's "2.16" is the 3-region AVERAGE; Center-only differs. Report 3-region,
  seed-averaged, val-selected once.

**Conventions** (so every run is analyzable & publication-ready):
- Each run writes to its own dir: `experiments/<expID>/.../{meta.json, preds_test.csv, preds_val.csv, epoch_log.csv}`.
- Disk-light: NO per-config model checkpoints during sweeps (SAVE_CKPT=0); only the
  final selected model is saved with a checkpoint. Metrics+preds are tiny.
- Reproducible: all training uses `repo/src/train/models.py` (shared, byte-identical);
  eval via `repo/src/eval/eval_lib.py` (reproduces paper 0.0000).
- Aggregate anytime: `python repo/src/eval/aggregate_experiments.py`.
- Filesystem has ~195 TB free; constraint is the ~6 concurrent-GPU QOS limit.

## FALLBACK BASELINE = OUR published ConvNeXt-TFT/LSTM (the article's spatial model)
The fallback is **our own** ConvNeXt-Tiny + TFT (and LSTM) from the article — the
2.16 result — NOT the competition. Its predictions are preserved in
`/home/weizyuv/Models Evaluations/Predictions VS Actuals/` and reproduce to **0.0000**
via `repo/src/eval/reproduce_metrics.py`. Overnight work is **upside-only**: adopt a
new variant ONLY if it beats our published ConvNeXt-TFT **on validation**; otherwise
ship the published model. Nothing overwrites it. (Tab-LSTM/SARIMAX/Prophet/TimesFM are
the competition we beat, not the fallback.)

## Baseline bar (Center, summer, MAE °C) — validated, reproduces paper 0.0000
| model | p1 all | p1 ext | soft | note |
|---|---|---|---|---|
| Prophet | 4.69 | 9.24 | 2.39 | classical |
| SARIMAX | 4.69 | 9.19 | 2.37 | classical |
| TimesFM 2.5 (0-shot) | 5.07 | 10.35 | 1.74 | foundational, **we beat it** |
| Tab-LSTM | 2.77 | 1.90 | 1.46 | non-spatial DL control |
| **ConvNeXt-TFT (ours)** | **2.16** | **1.48** | **1.27** | spatial |

## SPATIAL EVIDENCE — 3-REGION COMPLETE (a3b0, k1, seed-avg, moving-block bootstrap)
All runs apples-to-apples: same a3b0/k1 loss, same 240-row test splits, 3 seeds, val-avg selection.

**DECISION (author, 2026-06-23): KEEP ORIGINAL PAPER CLAIMS — fall back to the published
ConvNeXt-TFT. The mean-frame structure-isolation ablation below stays INTERNAL (out of the
paper). No overnight variant is adopted (none beat the incumbent on all 4 metrics).**

1. **Claim 1 — beats baselines (IN PAPER, PROVEN):** spatial ConvNeXt-TFT p1-all ~2.07 beats
   Prophet 4.69, SARIMAX 4.69, TimesFM-2.5 5.07, paper Tab-LSTM 2.77 — decisively (repro-sig p<0.001).
   vs strong re-tuned non-spatial DL controls (R4 answer): 3-region pooled 2.087 vs tab-LSTM 2.205,
   Δ+0.118 °C, p=0.13 (directional, not significant). Moirai (B2) pending to make foundational plural.
2. **Claim 2 — spatial STRUCTURE isolation (INTERNAL ONLY, mean-frame vs full, same TFT):**
   3-region pooled, seed-avg, bootstrap:
   - ALL-case: full 2.087 vs mean 2.056 -> Δ−0.031, p=0.57 (NULL; mean marginally better).
   - EXTREME: full 1.606 vs mean 1.644 -> Δ+0.038, p=0.65 (NULL).
   - Per region: spatial helps Center +0.23 (sig), HURTS Negev −0.10 & NW −0.22 -> cancels on average.
   - Better mechanism doesn't rescue: attn-pool 2.74 > GAP 2.19 (exp09). frozen-backbone 3.06 (FT matters).
   => Honest internal verdict: spatial structure helps only in heterogeneous (coastal/urban) Center;
      regional aggregate suffices in homogeneous desert regions. NOT a 3-region effect.
      Physically sound (daily Tmax within a ~100 km cluster is highly spatially correlated).
      Per author decision, this ablation is NOT reported in the paper; the published pipeline-level
      result (spatial method > non-spatial baselines) stands as the paper's spatial evidence.

## Experiment log
| expID | what | config | status | key result | dir |
|---|---|---|---|---|---|
| repro-eval | reproduce paper metric table | — | DONE | max\|Δ\|=0.0000 (hot+soft, 5 models) | results/summer_metrics_*.csv |
| repro-sig | significance (corrected pooling) | B=10k, block4/2, Holm | DONE | TFT vs Tab-LSTM hot +0.60 p<0.001 | results/summer_significance.csv |
| A6-pertarget | per-target p1/p3/p7/p15 MAE | — | DONE | spatial wins p1/p3; classical wins p15 | results/summer_per_target.csv |
| B1-timesfm | TimesFM 2.5 zero-shot (3 regions) | ctx1024,h30 | DONE | p1 all ~4.9 vs ours 2.16 (we win) | results/timesfm_*.csv |
| A3-kernel | interpolation EXP vs IDW | TFT k2a2b1 pat10 | DONE | EXP 2.08 < IDW 2.40 (exp wins) | experiments/exp02_frames/Center_{EXP,IDW}_tft |
| diag-patience | reproduce w/ correct patience | k1a1b0 pat30 | DONE | test 2.11 (pat10 gave 3.14) → pat30 reproduces | experiments/exp03_.../a1_b0 |
| A4-loss | loss ablation α×β (k=1,pat30) | TFT, 9 cfg | DONE | all-vs-extreme tradeoff; val-sel(avg)=α2β1 test 2.66/1.72; val_all-sel=α3β0 test 2.11 | exp03_center_hot_pat30/a*_b* |
| A2-window | input window 90/180/270/365 | TFT k1a2b1 pat30 | DONE | **180 optimal** (2.12); 90=2.32,270=2.57,365=2.76 | exp02_frames/A2_win* |
| exp04-arch | head comparison (NEW models) | ConvNeXt+{TFT,LSTM,PatchTST,iTransformer} k1a2b1.5 | DONE | **PatchTST val-best 2.28**; iTrans test 2.15; LSTM 2.42; TFT 2.37 | exp04_arch/<head> |
| exp05-kabl | k steepness ablation | TFT k∈{2,3} a2b1 (k=1 from A4) | DONE | **k=1 best** (k2 val2.86, k3 val2.97 > k1) | exp05_kabl/k* |
| exp06-lstm | LSTM loss ablation α×β | LSTM k1 9cfg pat30 | DONE | val-best a1b0 → test 2.10 (≈TFT) | exp06_lstm_loss/a*_b* |
| exp07-region | scale to Negev+NW | TFT k1a2b1.5 | DONE | Negev test 2.20, NW 2.43 (Center 2.37) → 3-reg avg 2.33 | exp07_regions/*_hot |
| exp08-seedavg | seed-avg best cfg/head (variance fix) | 4 heads × 3 seeds (111/222/444) | RUNNING | — | experiments/exp08_seedavg/* |
| exp09-pool | **representation: spatial pooling** (use frames properly) | ConvNeXt {avg(GAP,published), attn, avgmax} × 3 seeds, TFT k1a2b1 | RUNNING | — | experiments/exp09_pool/* |
| exp10-anom | **representation: day-of-year climatology anomaly channels** (extreme=deviation) | EXP+anomaly augment (18ch) | SMOKE | — | experiments/exp10_anomaly/* |

### Representation/architecture notes (goal: best overall MAE + track extreme/inter-annual trend)
- ConvNeXt-Tiny downsamples 44×137 by 32× → only ~2×5 (~10) spatial tokens before pooling.
  So GAP averages ~10 tokens; **avgmax** (mean+**max**) captures the hottest token → fits extremes.
- Anomaly channels feed deviation-from-DOY-climatology (train-only, no leakage) so the model
  sees "extreme = unusually high" directly instead of inferring season from raw °C.
- Pretrained weights search: weather FMs (ClimaX/Aurora/Prithvi-WxC) are 0.3–2.3B params built
  for global ERA5/MERRA-2 (160+ vars) — mismatched for our 9-ch 68-station regional frames;
  not a clean drop-in. ImageNet ConvNeXt-Tiny stays the practical pretrained backbone.
  (Future: regional fine-tune of a weather FM.)

## Planned next (queued behind GPU QOS)
- A4b: LSTM head loss ablation; k-ablation (k∈{1,2,3} at central α/β) → justify k.
- ARCH: temporal-head comparison ConvNeXt+{TFT, LSTM, PatchTST, iTransformer} at best loss cfg → justify head choice ("new models").
- ENS: TFT+LSTM ensemble (post-hoc, free).
- A1: target-set {1,3,7,15} vs {1,5,10,20}.
- T1.3: interpolation FIX (paper-accurate normalized kernel) — needs notebook regen or 3 grid params.
- Tier3: Moirai-MoE + tabular PatchTST/iTransformer (non-spatial) baselines.
- Tier4: scale to {Center,Negev,NW}×{hot,cold}; downstream degree-day proxy; repro appendix; vector figures; writing fixes (article_review_notes.txt).

## Reject-fix tracker (paper accuracy issues found)
- "R²≥0.5 feature selection" is mis-stated: prs_stn (R²=0.999) dropped; 3 kept feats <0.5.
  → reword to "temperature/humidity/heat-stress family; pressure excluded".
- Abstract "Wilcoxon" → should be moving-block bootstrap (no Wilcoxon table).
- Significance "ALL" pooling was 3× inflated (n2148→716) — corrected.
- Soft metric is abs-of-mean in summer but mean-of-abs in winter — unify + state.
- Fig 3 unreadable → replaced by clean vector p1 actual-vs-pred (results/figures/).
