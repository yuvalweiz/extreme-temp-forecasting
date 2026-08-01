# Resubmission Strategy — Extreme-Temperature Forecasting

Goal: get the method to **state-of-the-art on its own terms** and make the
paper bulletproof against the ECML-PKDD 2026 reviews — while keeping the core
method and the two hard constraints fixed:

- **Hard constraints (do not change):** 3 clusters; 1 country (Israel).
- **Core method (keep, may tune):** elevation-aware station→grid frames;
  k-value channel selection (R²-based); ConvNeXt-Tiny spatial backbone;
  TFT-Lite / LSTM temporal head; percentile-weighted extreme-aware loss
  (`k`, `alpha`, `beta`); month-ahead order-statistic targets (1st/3rd/7th/15th);
  strict chronological train/val/test split, no leakage.

The thesis we must defend: **spatial information adds predictive signal beyond
what any non-spatial model can extract from cluster history.** The cleanest
proof is to beat *every* non-spatial method — classical, tabular-DL, and
foundational — on identical splits, targets, and metrics.

---

## 1. Why it was rejected (3 pillars + secondary)

From the 4 reviews + meta-review (`rejects/Reviews_ECML2026.txt`):

| # | Pillar | Who | What they want |
|---|--------|-----|----------------|
| P1 | **Reproducibility** | R2,R3,R4,Meta | dataset size, splits, #samples/#features, full hyperparameters, model citations, **public code/GitHub**, an algorithm/flowchart |
| P2 | **Weak baselines** | R4,Meta | SARIMAX/Prophet are too weak/"controversial"; add **recursive, transformer-based, and foundational** forecasters |
| P3 | **Missing ablations / justification** | R1,R3,Meta | ablate & justify: 1st/3rd/7th/15th targets, 180-day window, interpolation bandwidth, loss weights, extreme weighting; justify the **Mean-MAE averaging** |
| S1 | Downstream impact | R1,R3 | quantify how °C MAE gains translate to demand/reserve planning |
| S2 | Presentation | R2,R3,R4 | Fig 1/2/3 low-res/overlapping; restructure Sec 3–4; cite base models |
| S3 | Per-region difficulty | R2 | discuss which regions are easier/harder |

Note the *positives* we must not lose: soundness rated **High** by R1,R2; the
elevation-aware gridding + order-statistic targets were explicitly praised; the
method "meticulously designed." We are strengthening, not rebuilding.

---

## 2. The winning narrative

> Month-ahead extreme-temperature order statistics for a demand region are
> better predicted from **nationwide spatial weather structure** than from the
> region's own history — and this holds against classical, tabular-deep, and
> foundation-model forecasters, with bootstrap significance.

Everything below serves that sentence.

---

## 3. Experiment plan — concern → experiment → justification

### P2 — Stronger baselines (highest priority; this is what flipped R4→reject)
All baselines are **non-spatial**: they see the same region-aggregated daily
series and predict the same order statistics on the same splits. If the spatial
model beats all of them, spatial signal is real.

- **B1 Classical (keep):** SARIMAX, Prophet — already done; keep as references.
- **B2 Tabular-DL (keep + extend):** Tab-LSTM (have). Add tabular PatchTST &
  iTransformer on the aggregated multivariate series (heads already in code).
- **B3 Recursive DL:** an autoregressive seq2seq LSTM that rolls daily temps
  then reduces to order statistics (addresses R4 "no recursive methods").
- **B4 Foundation models (zero-shot + light fine-tune):** TimesFM and Chronos
  on the daily series → derive the order statistics from the 30-day forecast
  (env is literally `timesfm311`). This is the single most important addition;
  it directly answers Meta + R4 "no foundational models."
- **Justification:** a like-for-like protocol (Appendix algorithm) for turning a
  time-series forecast into the 4 order statistics, identical for every model.

### P3 — Ablations (each is a clean, pre-registered sweep)
- **A1 Targets {1,3,7,15}:** sweep alternative anchor sets (e.g. {1,5,10,20})
  and the anchor-vs-soft choice; report sensitivity. Answers R1/R2/R3 "why p1/p3/p7/p15".
- **A2 Input window:** {90, 180, 270, 365} days. Answers R1/Meta "180-day not justified".
- **A3 Interpolation bandwidth (k / γ) & elevation term:** with/without elevation,
  sweep bandwidth. Answers R1 "interpolation bandwidth not justified" + R4 "how
  interpolation is used."
- **A4 Loss:** sweep `k` (percentile steepness), `alpha` (under-prediction),
  `beta` (hinge) — **this is the exp01 grid already running**. Answers R1 "loss
  weights / extreme weighting not ablated."
- **A5 Spatial vs tabular (the core ablation):** ConvNeXt+head vs identical head
  on cluster-mean series — already the headline; report per-region.
- **A6 Mean-MAE justification:** always report per-target (1/3/7/15) AND all-vs-
  extreme split separately, so the aggregate is never the only number. Answers
  R3/R4 "averaging is controversial."

### P1 — Reproducibility (this repo is the deliverable)
- Public GitHub repo = `repo/` (this). Modules: data → interpolation → frames →
  dataset → models → losses → train → eval → significance → baselines.
- Appendix tables: dataset stats (stations, period, #samples per split, #features
  + R² selection), full hyperparameters, exact architecture (with citations),
  and a method flowchart/algorithm box.
- `eval_lib.py` reproduces the published metric table to **0.0000** (verified).

### S1 — Downstream impact (lightweight, high payoff)
- A small degree-day / peak-load proxy: map predicted extreme °C → cooling/
  heating degree-days → an illustrative reserve-margin error. Even a simple,
  clearly-caveated linkage answers R1/R3 and the meta-review.

### S2/S3 — Presentation & per-region
- Regenerate all figures as vector PDFs (already partly in `Models Evaluations/
  Figures/*.pdf`); fix Fig 1 grids, Fig 3 overlap (plot residuals per R4).
- Add a per-region difficulty paragraph (Negev hottest/most-extreme vs NW).

---

## 4. Tuning levers to reach SOTA (core logic preserved, every change justified)

| Lever | Change | Why it's allowed / justified | Leakage guard |
|-------|--------|------------------------------|---------------|
| Interpolation fix | implement paper-accurate normalized-exp IDW + elevation; fix the magic-constant/normalization bug | "fix bugs" — the frames should match the *described* method | recomputed from raw obs only |
| Fresh data | rebuild frames from refreshed IMS pull (1920–2026) | more train + more test (helps R1 small-n_ext) | split by date, fit norm on train only |
| Loss `k/α/β` | exp01 sweep | A4 ablation | selection on val only |
| Architecture | head depth/width, bi-LSTM, TFT+LSTM ensemble | tuning within the same backbone+head family | val-selected |
| Seeds | 3-seed averaging | variance control (R1 n_ext caution) | n/a |

**Disk discipline:** sweeps write only metrics + tiny preds CSVs (no per-config
checkpoints); one checkpoint saved for the final selected config. No new frame
sets duplicated unnecessarily.

---

## 5. Status (live)

- ✅ Reviews + own notes parsed; pillars above.
- ✅ IMS data refreshed: 8.21M hourly rows, 1920–2026 (parquet verified).
- ✅ Eval+significance recovered and rebuilt → **reproduces paper table exactly
  (max abs diff 0.0000, hot+soft, all 5 models)**.
- ✅ Significance recomputed with **corrected single-region pooling** (was tripled:
  n=2148→716, n_ext=285→95). **Headline survives:** TFT vs Tab-LSTM hot
  all-case +0.60°C p<0.001, hot-extreme +0.35°C p=0.028.
- ⏳ exp01 Center-hot grid (repro + k/α/β ablation) queued on cluster (disk-light).
- ⏭ Next: confirm exp01 → full grid; interpolation fix; fresh-data frames;
  foundational baselines (TimesFM/Chronos).

## 6. Headline numbers to beat (summer, avg of 3 clusters, MAE °C)

| model (non-spatial unless noted) | all | extreme | avg |
|---|---|---|---|
| Prophet | 4.69 | 9.24 | 6.97 |
| SARIMAX | 4.69 | 9.19 | 6.94 |
| Tab-LSTM | 2.76 | 1.90 | 2.33 |
| **ConvNeXt-LSTM (spatial)** | 2.34 | 2.10 | 2.22 |
| **ConvNeXt-TFT (spatial)** | **2.16** | **1.48** | **1.82** |

Target for the resubmission: keep/extend this lead **and** beat TimesFM/Chronos
and tabular-transformer baselines on the same protocol.
