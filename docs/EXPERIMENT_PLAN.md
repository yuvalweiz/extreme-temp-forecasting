# Experiment plan — what we must validate to answer the reviews

Order follows the user's directive: **maximize our model first, then ablations,
then strong baselines.** Everything is on Center/hot first, then scaled to
{Center, Negev, Northwest} x {hot, cold}. Reproducible: all runs use the same
`models.py` + either the sample grid (`train_grid_hot.py`) or the frames trainer
(`train_from_frames.py`).

Key efficiency unlock: `train_from_frames.py` makes the **window (A2)**,
**interpolation kernel (A3)**, and **target-set (A1)** ablations parameter
changes on ONE pipeline — no new datasets to build.

## TIER 1 — fix / maximize our model  (do first)
| id | experiment | answers | status |
|----|-----------|---------|--------|
| T1.1 | loss ablation k x alpha x beta (54 cfg) | A4 (R1) + best config | RUNNING (exp01) |
| T1.2 | interpolation kernel: EXP vs IDW vs EXP_V2 (on-disk frames) | A3 part 1 (R1,R4) | infra ready (exp02) |
| T1.3 | interpolation FIX: paper-accurate normalized-d3d frames | A3 part 2 (the bug fix) | needs pixel_matrix or regen |
| T1.4 | model tweaks: TFT+LSTM ensemble, bi-LSTM, seed-averaging | SOTA push + variance | pending |

## TIER 2 — design-choice ablations the reviewers demanded (P3)
| id | experiment | answers | cost |
|----|-----------|---------|------|
| A2 | input window {90,180,270,365} | "180-day not justified" (R1,meta) | trivial w/ frames trainer (HISTORY param) |
| A1 | target set {1,3,7,15} vs e.g. {1,5,10,20} | "why p1/p3/p7/p15" (R1,R2,R3) | cheap (re-derive y, retrain) |
| A6 | report per-target (1/3/7/15) AND all-vs-ext separately | "averaging controversial" (R3,R4) | eval-only |
| A5 | spatial vs Tab-LSTM (held) | core claim | have it; extend per region |

## TIER 3 — stronger baselines (P2 — the reject driver from R4 + meta)
| id | baseline | protocol | status |
|----|----------|----------|--------|
| B1 | TimesFM 2.5 (zero-shot + light FT) | forecast 30d -> sort -> [0,2,6,14] | env ready (timesfm) |
| B2 | Moirai-MoE | same | env ready (uni2ts) |
| B3 | tabular PatchTST / iTransformer | direct order-stat regression | heads in models.py |
| B4 | recursive seq2seq LSTM | roll daily temps -> reduce | small build |
| -- | (keep) SARIMAX, Prophet, Tab-LSTM | already done | done |
Must BEAT all of these to defend "spatial info matters."

## TIER 4 — scale, downstream, reproducibility, writing
- Scale every Tier-1/2 result to {Center,Negev,Northwest} x {hot,cold}.
- S1 downstream: cooling/heating degree-day -> reserve-margin proxy (R1,R3).
- P1 reproducibility appendix: dataset stats (#stations, period, #samples/split,
  #features+R2 selection), full hyperparameter table, method flowchart, GitHub link.
- S2 figures: regenerate as vector PDFs; fix Fig1 grids / Fig3 overlap (plot residuals).
- S3 per-region difficulty paragraph (Negev hottest/most-extreme vs NW).
- Writing fixes (article_review_notes.txt): abstract em-dashes; "Wilcoxon"->bootstrap;
  paste full significance table; fix conclusions table refs; caption typo.

## Compute/disk discipline
- Frames trainer preloads ~7.5k frames (~270s) then trains in RAM (1.3GB), no new
  disk. Ablations = many GPU-hours -> batch sbatches, save metrics+preds only.
- Interpolation FIX (T1.3) is the only step that writes new frames (~2.2GB, new dir).

## Open item that gates T1.3 (interpolation fix)
Need the exact generation grid. Cleanest: user saves `pixel_matrix` (one np.save)
from the run that made `EXP 05_01 ... HIGHRES`. Without it, T1.3 falls back to a
self-consistent regen (my grid, my-exp vs my-paper) — still a valid ablation.
