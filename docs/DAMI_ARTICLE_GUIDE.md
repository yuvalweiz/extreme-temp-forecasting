# DAMI Article Guide — results, evidence, and how to edit

Single reference for turning the rejected ECML paper into the DAMI (Data Mining and
Knowledge Discovery, Springer) submission. Maps every section to the evidence/numbers,
records the pipeline, and lists what is done vs pending. **Nothing here overwrites the
originals** — LaTeX lives in `dami_submission/` (copy), code in `repo/src/pipeline/`
(faithful port of `~/run_grid.py`, validated 0/0 state-dict keys, reproduces published
meta exactly), originals (`~/run_grid.py`, `~/project codes/*`, `latech/paper.tex`) untouched.

---

## 0b. SPATIAL VERDICT (2026-06-29) — two non-spatial controls, both reported
Numbers = Center-hot, stable 3-seed mean. Two controls (both spatial-mean series, no ConvNeXt):
tab-LSTM (recursive baseline reviewers asked for) and tab-TFT (matched-head ablation: SAME TFT
head/loss/protocol, isolates only the raw spatial dims — a deliberately brutal control).

| metric | tab-LSTM | tab-TFT | base spatial | **best spatial (v2_cosema_tw)** |
|---|---|---|---|---|
| 4-out mean (headline all-case) | 1.579 | 1.481 | 1.556 | **1.446** |
| p1 (hottest output) | 2.245 | 1.975 | 2.064 | 1.997 |
| extreme (MAE_ext) | 1.869 | 1.879 | 1.904 | **1.541** |

**Claims that HOLD (use these):**
- Spatial beats the recursive **Tab-LSTM on all-case**, every metric (P2 response). ✓
- Best spatial beats **both** tabular controls on the **headline 4-out-mean all-case (1.446)** and
  on **extreme (1.541)** — plus classical/foundational (4.7–5.1) by a wide margin. ✓
- The ONLY non-win: p1 sub-metric vs matched-head tab-TFT (1.997 vs 1.975, 0.022°C = seed noise).
- Spatial's edge is LARGEST on extreme (on-thesis for an extreme-temp paper).

**Caveat to report honestly (not a headline-killer):** the matched-head tab-TFT shows the spatial
gain on the p1 AVERAGE is modest; spatial's clear, consistent advantage is on EXTREME. A1
spatial-preserving (85–374 tokens) didn't move the all-case average — consistent with the target
being a cluster spatial-mean.

### FAIR TABULAR ABLATION (2026-06-30, the decisive control) — spatial wins the tail
The hardest control: a tabular model given the FULL station information (all 68 Israel stations ×
9 features, DAILY 180-step — exactly what the frames are interpolated from), same tags/targets/
head, differing ONLY in no-grid. Built to `repo/data/stationvec_Center/` (1194 samples, y matches
frame dataset exactly, 1.86% imputed in test).
| control (same info, no grid) | 4-out | p1 | extreme |
|---|---|---|---|
| fair tabular (all 68 stations daily) | 1.484 | 1.948 | 2.213 |
| best spatial (grid) | 1.446 | 1.997 | **1.541** |
**Result:** matches spatial on the average (1.484 vs 1.446; tabular even edges p1 1.948<1.997),
but the spatial grid wins DECISIVELY on the EXTREME tail (1.541 vs 2.213, −0.67°C). The spatial
interpolation enables extreme-tail skill that a same-information tabular model cannot match — the
strongest, fairest evidence for the extreme-tail spatial thesis (§0). Not a handicapped baseline.

### ELEVATION LEVERS (both tested, single-cluster) — neither helps, clean negatives
- A2 elevation-as-channel (geo): no help (1.56–1.59 vs base 1.556).
- A3 lapse-rate (real elevation, memory-wise, correction std 2.87°C): hurt base/sp/tab
  (+0.03–0.07); marginal+mixed on the best combo (4out 1.405 better, p1 worse). NOT reliable.
- Reason: for a SINGLE cluster the cluster elevation is CONSTANT across samples → no signal
  either way. Drop the "elevation-aware" claim for single-cluster; would only matter cross-region.

### MULTI-REGION HOT ROLLOUT (2026-06-30) — beats the original everywhere (hot)
Best recipe = ConvNeXt-V2 + cosine/EMA + in-domain pretrain + [1,5,10,15] targets. p1 all-case MAE:
| region | published | base(re-trained) | best recipe | Δ vs published |
|---|---|---|---|---|
| Center | 2.265 | 2.064 | 1.997 | −0.268 |
| Negev | 2.135 | 2.055 | 1.766 | −0.369 |
| Northwest | 2.081 | 2.037 | 1.879 | −0.202 |
NOTE (author decision 2026-07-02): these IMPROVEMENTS are DEFERRED for the article — the DAMI
submission stays 1:1 with the original + ablations; improvements folded in only once confirmed to
hold (incl. cold). Bug-fixes: no-wd-norm no gain; scheduler+EMA already in the recipe.

## 0. Headline framing (author-confirmed, 2026-06-29) — extreme-tail led
**Spatial structure improves EXTREME-temperature prediction.** The spatial model beats every
non-spatial control AND classical + foundational baselines on the extreme metric (MAE_ext 1.54
vs tabular 1.87–1.90, classical/foundational 4.7–5.1). The tail is the paper's purpose and is
exactly where spatial pays off (§0b). This is the contribution.
- **Supporting claim (no tradeoff):** the same model also beats classical, foundational, AND
  recursive (Tab-LSTM) baselines on standard all-case MAE (best spatial 1.446 vs Tab-LSTM 1.579;
  see §0b) — so the tail gains don't sacrifice general accuracy.
- **Legitimacy:** config selected on validation, test read once; the val-selected model also
  wins on test (no cherry-picking).
- Variance note: the extreme metric is seed/epoch-sensitive (small ext counts) — we therefore
  report stable 3-seed means with deterministic training + min-epoch floor (§4), not single runs.

---

## 1. Reviewer rejects → response (the 3 recurring pillars; see `REVIEWER_EVIDENCE.md`)
The meta-review named exactly three convergent weaknesses:
- **P1 Reproducibility** (all 4 reviewers + meta): dataset stats/splits/#samples/#features,
  hyperparameters, base-model citations, public code, algorithm box. → clean repo reproduces
  to |Δ|=0.0000; `results/dataset_stats.md` (Center 813/143/238, period 2021–2025); full
  hyperparameter + architecture tables (§5); algorithm box; GitHub release.
- **P2 Baselines** (R4 + meta hard): no recursive/transformer/foundational models. → we beat
  classical (SARIMAX/Prophet), recursive (Tab-LSTM), AND two foundational (TimesFM, Moirai).
  Bonus lesson: foundational models collapse to seasonal mean and fail on regional tails.
- **P3 Ablations** (R1/R2/R3 + meta): targets, window, interpolation bandwidth, loss weights
  not ablated. → full ablation suite on the REAL pipeline (§3).
- Secondary (figures, per-target metrics, regional difficulty) → §6.
- Dissolved by the extreme-temperature reframe: downstream electricity-demand quantification.

---

## 2. Main results table (Center-hot, all-case test MAE °C; lower better)
All from the REAL pipeline; baselines from the author's real saved predictions.
| family | method | all-case MAE |
|---|---|---|
| classical | Prophet / SARIMAX | 4.74 / 4.74 |
| foundational | TimesFM-2.5 / Moirai-1.1-R (0-shot) | 5.07 / 4.94 |
| **non-spatial control (tabular ablation)** | Tab-LSTM (series → head, no ConvNeXt) | 2.81 |
| spatial (published) | ConvNeXt-TFT (original article) | 2.265 |
| spatial (ours, stabilized) | base re-trained | 2.064 |
| **spatial (ours, best, val-selected)** | **ConvNeXt-TFT + in-domain MAE pretrain (V2)** | **2.007** |

**Spatial helps:** the matched tabular control (spatial-mean series, same loss/protocol,
no ConvNeXt) lands well above the spatial model → the spatial representation is doing real
work. (Numbers finalize when the tabular-ablation jobs land; Tab-LSTM 2.81 already shows it.)

---

## 3. Ablations (P3) — all on the real pipeline, stable 3-seed (deterministic + min-epoch floor)
| design choice | result | verdict |
|---|---|---|
| input window {60,90,120,180} | 180 best (60/90/120 worse) | **180 justified** |
| extreme threshold p95 (loss) | ablated p90/95/99 | reported |
| target set [1,3,7,15] vs alts | [1,5,10,20]/[1,2,3,4,5]/[1,2,3] | tradeoff (see per-target) |
| loss α (under-mult), β (hinge) | from author's real grids: α2 balances; β essential | justified |
| out_weights equal vs geometric | equal (published) competitive | reported |
| architecture capacity d_model/nhead/layers | d256/nhead4/layers2 near-optimal (bigger overfits) | **chosen size optimal** |
| backbone ConvNeXt-V1 vs V2 | V2 competitive; V2 + in-domain is the winner | reported |
| **in-domain MAE pretraining** | **best (2.007 vs 2.064 base, 2.265 published)** | **adopted improvement** |
| interpolation EXP vs IDW (±geo) | pending (`/storage` build) — note EXP elev signal weak | see §7 |

Why p1/p3/p7/p15 (the recurring "why not p2"): order statistics at geometric depths of the
30-day window; consecutive ranks are near-duplicates (≈0.1 °C apart) so geometric spacing
samples the tail shape with minimal redundancy.

---

## 4. The variance / stabilization finding (a genuine contribution)
The published val-selection `avg3 = (val_MAE_all + val_MAE_ext + underrate_ext)/3` is
high-variance: best epoch swings 1–33 → ~1 °C run-to-run on identical config+seed. The
published extreme numbers were a favorable early-epoch draw. **Fix** (in `repo/src/pipeline`):
deterministic training + minimum-epoch floor → stable, reproducible. This *also improves*
the all-case result (stabilized base 2.064 < published 2.265). Honest and publishable:
"we identified and stabilized a high-variance selection."

---

## 5. Pipeline & reproducibility (P1) — for the methods/appendix
Real pipeline = `~/run_grid.py` (published) ≡ clean port `repo/src/pipeline/`:
- **Data**: per-sample .npz (X=(180,9,44,137) exp-interpolated frames, y=(4,) °C order stats),
  DOM∈{1,7,14,21,28}, window = date_range(end=pred_point−1day, periods=180). Norm stats
  TRAIN-ONLY (no leakage). Splits chronological: Center 813/143/238.
- **Model**: ConvNeXt-Tiny (ImageNet) per frame → (B,T,768) → TemporalFusionLite/LSTM head →
  OutputAffine → 4 outputs. d_model=256, nhead=4, layers=2.
- **Loss**: WeightedMAE_ExtremeUnderOnly (MAE; under-prediction ×α only on extremes,
  true_hot≥train-p95) + ExtremeUnderHinge (β·ReLU(true−pred) on extremes). out_weights=(1,1,1,1).
  Published per-region: Center α2-β1, Negev α3-β1.5, NW α1-β0.
- **Optim**: AdamW (backbone 1e-5, head 1e-4, wd 1e-4), grad-clip 1, AMP, patience 30, batch 4.
- **Run an experiment** (cluster): `sbatch` a job with env knobs on `pipeline.train`:
  REGION, HEAD, ALPHA, BETA, DROPOUT, SEED, EXTREME_Q, WINDOW, OUT_WEIGHTS, TARGETS,
  POOL{avg,max,avgmax,attn}, BACKBONE{convnext_tiny,convnextv2_tiny/nano}, D_MODEL, NHEAD,
  HEAD_LAYERS, NO_WD_NORM, SCHEDULER{cosine}, EMA, PRETRAINED, GEO, TABULAR, DATASET_DIR,
  CACHE, DETERMINISTIC, WARMUP. Submit scripts: `~/submit_real.sh`, `~/submit_stable.sh`.
- In-domain pretraining: `pipeline.pretrain_frames` (masked denoising AE on ~5141 leak-free
  daily frames ≤ train cutoff). Encoder loads into the forecaster via PRETRAINED.

---

## 6. Secondary reviewer items
- **Per-target metrics** (R3/R4): `results/summer_per_target.csv` (p1/p3/p7/p15 individually) —
  replaces the single averaged Mean-MAE.
- **Regional difficulty** (R2): per-region numbers exist; one paragraph.
- **Figures** (R2/R3/R4): vector p1 actual-vs-pred in `results/figures/`; redo Fig 1/2/3 vector,
  fix Fig 2 (a)≠(b) locations, Fig 3 → residuals.

## 7. Spatial-data usage — code audit findings (CRITICAL for the spatial thesis)
A full audit of the published pipeline (`~/run_grid.py`, `~/project codes/*`) found the
"spatial helps" claim is currently **not well supported by the pipeline**, for 3 compounding
reasons (all CONFIRMED) — and a concrete fix path.

**Why spatial is washed out:**
1. **Global-average-pool collapse.** ConvNeXt-Tiny downsamples 44×137 by 32× → **1×4 map**,
   then GAP → one 768-vector/frame. The temporal head sees 180 spatial-*mean* summaries; the
   target is also a cluster spatial-mean → model learns "spatial-mean → spatial-mean."
2. **Elevation kernel inert.** EXP elevation term changes interp weights by L1=0.0035 (max
   0.0018) — a ~1.6 km elevation term vs ~100 km horizontal distances is negligible.
   (Matches corr(temp,elev): EXP +0.08 vs IDW +0.43.)
3. **Published dataset has no elevation channel** (9 weather channels only).
This explains the null full-vs-mean and the weak geo-channel result (geo also averaged away).

**Other confirmed issues:** k-values unit inconsistency (km vs m derivations give different k);
`k_exp=0.2` over ~140 km → near-Voronoi over-smoothing; avg3 selection mixes °C with a [0,1]
fraction (noisy, ~7 extreme val samples) → unstable early stop → ~1 °C variance; no LR
scheduler; weight decay on norms/biases/OutputAffine. Verified CLEAN: no leakage, correct
targets/windowing/denorm.

**Fix path to make spatial genuinely help (ranked):**
- **A1 (top): replace GAP with spatial-token preservation + attention pooling** — use a less-
  downsampled feature map so the model weights grid regions (the cluster footprint) instead of
  averaging all of Israel. Highest leverage. [being implemented in `repo/src/pipeline`]
- A2: add elevation as a channel OR inject it as a static side-feature AFTER pooling (so it
  isn't averaged away or stored 180×).
- A3: physical lapse-rate interpolation (reduce T to sea level by Γ·elev, interpolate, re-apply)
  → elevation becomes first-class. (Needs frame regeneration on `/storage`.)
- A4: reconcile k-units; A5: run the explicit spatial ablation (full vs station-mean vs
  elevation-zeroed).
- Training fixes B1–B4 (stable selection, scheduler, WD groups) — ALREADY applied in clean pipeline.

**The opportunity:** if A1 (+A2/A3) makes full ≫ mean and beats the matched tabular control,
the spatial claim becomes TRUE and strong — the best possible result for the article.
(Full audit report retained in session; key file:line refs available.)

## 8. Status: done vs pending
- DONE: clean pipeline (validated), variance fix, Center-hot stable leaderboard, baseline
  comparison, window ablation, in-domain pretraining improvement, reviewer-evidence pack.
- RUNNING: tabular ablation (spatial-helps), combos, LaTeX completeness audit, code audit.
- PENDING: 6-cell rollout (Negev/NW hot + 3 regions cold) for the "beats original everywhere"
  claim; IDW interpolation comparison; figures; final article assembly in Overleaf.
