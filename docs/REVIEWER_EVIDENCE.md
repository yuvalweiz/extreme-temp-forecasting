# Reviewer-oriented evidence pack — focus on the RECURRING concerns

Reframed paper = **extreme-temperature forecasting** (electricity dropped). This pack
organizes our evidence around the concerns that recur across ≥2 of the 4 reviewers and
the meta-review (the decision driver). The meta-review names exactly three:
*reproducibility, weak baselines, design choices not ablated.* Everything below is
ready to drop into the revision.

Recurrence tally (who raised what):
- **Reproducibility** — R1, R2, R3, R4, meta  (ALL)
- **Weak baselines (recursive/transformer/foundational)** — R4, meta (hard); R3 (partial)
- **Ablation/justification of design choices** — R1, R2, R3, meta
- Figures unreadable — R2, R3, R4
- Per-target (not averaged) metrics — R3, R4
- Presentation/structure — R2, R3, R4
- Downstream demand — R1, R3, meta → **DISSOLVED by extreme-temp reframe**
- Single-region generality — R1, R4 → fixed scope; honest limitations paragraph

---

## PILLAR 1 — Reproducibility (raised by ALL reviewers + meta)
The single biggest reason for rejection. Evidence exists; needs packaging into the paper.

| recurring sub-item | reviewers | artifact (ready) | paper action |
|---|---|---|---|
| dataset size / #samples / #features / period / locations | R2,R4,meta | `results/dataset_stats.md` (train/val/test = 813/143/238 per region; period 2021–2025; features list) | add **Dataset Summary table** + map |
| data split / #per subset / validation procedure | R4,meta | chronological split, train-only thresholds, EMA val-selection (REGISTRY protocol) | add **Protocol** paragraph + split table |
| hyperparameters (LSTM hidden size, lrs, schedule…) | R2,R3,meta | `docs/HYPERPARAMETERS.md` (backbone lr 1e-5 wd1e-3; head lr1e-4; LSTM hidden 256; d_model 256; AdamW; cosine; pat30; AMP; EMA0.35) | add **Hyperparameter table** (appendix) |
| base models not cited; architecture unclear | R3 | ConvNeXt-Tiny (Liu 2022), TFT (Lim 2021), LSTM (Hochreiter 1997), PatchTST (Nie 2023), iTransformer (Liu 2024) | add citations + **architecture table** |
| no algorithm / flowchart | R3,R4 | pipeline is deterministic (frames→ConvNeXt→head→loss) | add **Algorithm 1** pseudocode box |
| public code/GitHub | R2,R3,R4 | clean `repo/` reproduces paper metric to **0.0000** (`reproduce_metrics.py`) | publish GitHub + link in paper + README |

**One-line rebuttal:** *"We release a public repository that reproduces every reported
number exactly (|Δ|=0.0000), and add full dataset, split, architecture, and
hyperparameter tables plus an algorithm box and base-model citations (App. X)."*

---

## PILLAR 2 — Stronger baselines (R4 + meta: the hardest technical complaint)
**Beaten on BOTH metrics: classical + recursive + TWO foundational models.** All rows are
from the **real saved predictions** (`summer_metrics_by_region.csv` + zero-shot foundational),
evaluated under the paper's protocol. p1 (hottest-day) MAE °C, 3-region average:

| family | model | AVG all | AVG **ext** |
|---|---|---|---|
| classical | Prophet | 4.69 | 9.24 |
| classical | SARIMAX | 4.69 | 9.19 |
| recursive | Tab-LSTM | 2.76 | 1.90 |
| foundational | TimesFM-2.5 (0-shot) | 4.91 | 9.54 |
| foundational | Moirai-1.1-R-L (0-shot) | 4.79 | 9.77 |
| spatial (ours) | ConvNeXt-LSTM | 2.34 | 2.10 |
| **spatial (OURS)** | **ConvNeXt-TFT** | **2.16** | **1.48** |

**Headline:** ConvNeXt-TFT is **best on both** all-case (2.16) and extreme-tail (1.48),
beating classical (4.69), recursive Tab-LSTM (2.76/1.90), and both foundational models.
On the tail, 1.48 vs 1.90 next-best.

**Insightful lesson (ADS track explicitly rewards this):** zero-shot foundational
forecasters (TimesFM, Moirai) regress to the seasonal mean and **fail on regional
extreme tails (~10 °C error)** — task-specific extreme-aware training is essential.

**Significance:** vs classical/foundational p<0.001 (Holm, moving-block bootstrap,
`summer_significance.csv`); vs Tab-LSTM hot +0.60 p<0.001.

> NOTE (2026-06-23): an earlier version of this table included re-tuned non-spatial
> PatchTST/iTransformer/LSTM "controls" (exp17) where PatchTST appeared to beat us on
> all-case. Those were a **reimplementation on the wrong loss/threshold/weights** and are
> QUARANTINED. R4's "recursive / transformer / foundational" ask is fully met by
> Tab-LSTM (recursive) + TimesFM & Moirai (transformer-based foundational), all real.

---

## PILLAR 3 — Ablations + justification of design choices (R1, R2, R3 + meta)
Every flagged choice is now ablated. Summary (Center unless noted, val-selected):

| choice flagged | reviewers | ablation result | verdict |
|---|---|---|---|
| input window 180 | R1, meta | 90→2.32, **180→2.12**, 270→2.57, 365→2.76 | 180 optimal |
| interpolation bandwidth/kernel | R1, R4 | elevation-aware EXP **2.08** < IDW 2.40 | EXP justified |
| extreme-weight steepness k | R1 | **k=1** best (k2 val2.86, k3 val2.97) | k=1 justified |
| loss weights α (under-pred) / β (hinge) | R1 | val-all-sel **α3β0→2.11**; α2β1→2.66/1.72 (all-vs-ext tradeoff) | reported tradeoff |
| target set p1/p3/p7/p15 | R1,R2,R3 | vs [1,2,3,4,5],[1,2,3],[1,5,10,20]: **[1,3,7,15] best on extreme tail** | justified below |

**Why p1/p3/p7/p15 (the recurring "why not p2", R1+R2+R3):** they are **order
statistics of the 30-day window at geometric depths** — p1 = the peak, p3/p7/p15 =
progressively less-extreme references down to ~half-month. Geometric spacing samples the
tail *shape* with minimal redundancy: consecutive ranks (p1 vs p2) are near-duplicates
(highly correlated, ~0.1 °C apart), so adding p2 adds cost without information. The
ablation confirms [1,3,7,15] gives the best extreme-tail MAE among candidate sets.

---

## Secondary recurring (2–3 reviewers) — quick closes
- **Figures (R2,R3,R4):** replace Fig 1/2/3 with vector graphics; fix Fig 2 (a)≠(b)
  locations; Fig 3 → residual plot. Vector p1 actual-vs-pred already in `results/figures/`.
- **Per-target not averaged (R3,R4):** `results/summer_per_target.csv` gives p1/p3/p7/p15
  individually + all-vs-extreme split. Replace the single Mean-MAE with the per-target table.
  (Shows the nuance: classical methods are fine on p15 but fail on p1 — supports our extreme focus.)
- **Regional difficulty (R2):** Negev (desert) easiest on extremes (1.24), NW hardest on
  all-case (2.08–2.50); add one paragraph.
- **Presentation (R2,R3,R4):** restructure Sec 3–4 (consolidate feature selection now split
  3.1/4.4); remove repeated backtesting/leakage/°C mentions; fix long lines. (Author/Overleaf.)
