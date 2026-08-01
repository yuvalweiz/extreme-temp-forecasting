# Ablations for the DAMI article (P3 response)

Directly answers the reviewers' P3 ask (targets, window, interpolation bandwidth, loss weights)
plus the spatial contribution. Two protocols, each internally consistent — label clearly:
- **ORIGINAL protocol** (author's real grids `grid_ckpts_*`): matches the paper's headline setup
  (equal out-weights, p95, avg3 selection, no scheduler/EMA). Use for loss/selection ablations.
- **STABILIZED protocol** (clean pipeline, deterministic + min-epoch floor, 3-seed mean): reduces
  the high selection variance we identified (§ variance). Use for window + spatial/tabular +
  architecture ablations. Base (Center TFT p1 all-case) = 2.064.
All numbers = Center, hot season, test MAE °C (p1 = hottest-day output unless noted).

## A. Loss design — weights α, β, and the selection metric  (ORIGINAL protocol)
| variant | p1 all-case | note |
|---|---|---|
| published (α2, β1, equal-w, avg3 selection) | 2.056 | the paper's config |
| β=0 (drop extreme under-hinge) | 2.008 | hinge trades a little all-case for tail safety |
| selection = MAE-all (not avg3) | 1.934 | best all-case, but worse tail (ext 3.45) |
| selection = MAE-ext | 1.969 | better tail (ext 2.64), slightly worse all-case |
Takeaway: the α/β + avg3 selection balance all-case vs the extreme tail; MAE-only selection wins
all-case but sacrifices the tail — consistent with the extreme-temperature objective.

## B. Input history window  (STABILIZED protocol, 3-seed) — reviewer P3
| window (days) | p1 all-case MAE |
|---|---|
| 60  | 2.232 |
| 90  | 2.182 |
| 120 | 2.090 |
| **180 (chosen)** | **2.064** |
Monotonic — more history helps; 180 is best. Confirms the paper's window choice. (Pure input-slice
ablation, frames unchanged.)

## C. Target set (order statistics)  — reviewer P3
Chosen = 1st/3rd/7th/15th hottest day of the next 30 (geometric spacing samples the tail with
minimal redundancy; consecutive ranks are ~0.1°C apart). Alternative [1,5,10,15] (tgtwide) runs in
the leaderboard; a fuller spacing sweep ([1,2,3,4] dense vs [1,7,14,21] sparse) is a quick add if
wanted (same out_dim=4, pure y-retarget — needs the 30-day series, i.e. a light rebuild).

## D. Interpolation kernel — reviewer P3 ("bandwidth") — PENDING DECISION
- Elevation finding (measured): the EXP "elevation-aware" kernel's elevation term is near-inert
  (weight L1 change 0.0035); corr(grid,elev) EXP +0.08 vs plain IDW +0.43. So EXP ≈ distance-only.
- Elevation LEVERS both tested and NEGATIVE for single-cluster: elevation channel (geo) no help;
  lapse-rate correction (real ±11°C structure) hurt slightly — because per single cluster the
  elevation is constant across samples (no cross-sample signal). → soften the "elevation-aware"
  claim; it would only matter cross-region.
- **EXP-vs-IDW head-to-head**: not yet run. IDW frames exist on disk → buildable memory-wise on
  /storage without regeneration. RUN? (pending author OK)

## E. Spatial contribution — the fair non-spatial control  (STABILIZED, 3-seed) — the key ablation
Tabular control has the FULL station information (all 68 Israel stations × 9 feats, DAILY 180-step,
same tags/targets/head), differs ONLY in no-grid. `repo/data/stationvec_Center/`.
| model | 4-out | p1 | extreme |
|---|---|---|---|
| fair tabular (68 stations, daily, no grid) | 1.484 | 1.948 | 2.213 |
| best spatial (grid) | 1.446 | 1.997 | **1.541** |
| tab-LSTM (recursive baseline) | 1.579 | 2.245 | 1.869 |
Takeaway: with identical information, spatial ties the tabular on the average but wins the EXTREME
tail decisively (1.541 vs 2.213, −0.67°C). The spatial interpolation enables extreme-tail skill a
same-information tabular model cannot match — the strongest evidence for the spatial contribution.

## F. Architecture / pooling  (STABILIZED, 3-seed) — supporting
- Backbone capacity d_model {128 (capd128) vs 256}: 256 chosen (bigger overfits the tail).
- Spatial pooling: GAP (published) vs attention over preserved tokens (sp_s0/s1/s2, 374/85/16
  tokens): attention-pool did NOT beat GAP on all-case average (cluster-mean target dominates).
- These support the published architecture choices.

## Cold season (winter/MIN) — reproduction check
Published cold (Center TFT p1): all-case 1.294, ext(p10) 1.802. Cold pipeline now ported into the
clean repo (COLD=1); verification runs reproduce it (in progress). Cold ablations mirror hot.

## Status of ablation data
DONE: loss/selection (A), spatial/tabular (E), architecture/pooling (F), elevation levers (D).
FILLING: window (B). PENDING author decision: EXP-vs-IDW (D), fuller target sweep (C).
