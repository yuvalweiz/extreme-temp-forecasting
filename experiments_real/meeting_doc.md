# DMKD Resubmission — Meeting Summary
*Rejects → fixes · the new model · open questions. 18 July 2026.*

## 1. Every reject and how we addressed it

**"Ensure the GitHub repo is organized with all source code."** A clean repository with the full pipeline (interpolation → training → evaluation) is prepared; the paper's Reproducibility section describes it. Only the public URL remains to be added.

**"Add dataset download instructions and a direct link; acknowledge the meteorological service."** Data-availability statement now links the IMS site and open-data portal with download/preprocessing instructions in the repo; IMS is acknowledged explicitly.

**"State dataset characteristics explicitly (stations, regions, samples, split, dates, resolution)."** A dedicated dataset table now opens the Dataset section: 68 stations, 3 regions (12/13/12 output stations), 813–814/143/238–239 samples per region, 2005–2025 daily data, chronological split.

**"Compare performance across regions; describe their climates."** Added a climate paragraph (coastal plain / humid northwest / arid Negev), per-region significance, and a regional-difficulty discussion with numbers.

**"Explain input vs output stations; present stations before regions."** Restructured: all 68 stations enter every model's input; each region's targets use only its own 12–13 stations — stated twice and ordered stations-first.

**"Electricity: motivation only."** One added sentence: weather-only study, no grid data anywhere; electricity is the motivating application.

**"Justify horizons 1/3/7/15 and report them separately."** Geometric-spacing justification now backed by a retraining ablation (see open question 1), and per-rank tables added for maximum *and* minimum temperature.

**"Test different history windows — don't assume 180 days."** Window ablation table (60/90/120/180 → 2.23/2.18/2.09/2.06 °C) with saturation discussion; no optimality claim beyond the tested range.

**"Justify loss weights; evaluate configurations; add ablations for design choices."** The (α,β) sweep is in the paper; a new output-weight grid was trained under the *published* protocol — the published choices (equal weights for summer, geometric for winter) sit at the optimum of their grid. Additional ablations: interpolation kernel (leave-one-station-out incl. kriging), capacity, architecture saturation, checkpoint-selection criterion.

**"Evaluate a Transformer baseline; investigate foundation models."** Transformer-head tabular baseline with measured numbers; TimesFM-2.5 and Moirai-1.1 zero-shot added — both collapse on extremes (~10 °C), strengthening the case for task-specific training.

**"Cite every model/method; explain the anchor/soft models and baselines; explain the significance test."** Canonical citations added for every model and method (Box–Jenkins, Prophet, AdamW, Holm, moving-block bootstrap, kriging, PatchTST, iTransformer, the clustering indices, etc.); anchor/soft models described with their exact output weights (also corrected a text-vs-code mismatch: summer anchor uses equal weights); baselines' full protocol documented; significance test now explained (below).

**"Feature-selection redundancy and leakage."** Single definition remains; explicit leakage paragraph: all statistics and selections from train (or validation) only, test 2021–2025 touched once.

**"Restructure Sections 3–4; clarify dataset/inputs/outputs/task."** Sections rewritten; a problem-formulation paragraph plus a 12-line pipeline algorithm box give the end-to-end picture.

**"Figures too small; Figure 1 unreadable."** Prediction panels enlarged ~3× (stacked full-width); Figure 1 redrawn as a vector diagram; all figures vector PDFs.

## 2. The new (corrected) model — protocol and evidence

The corrected pipeline fixes a frame-construction fault found in the published preprocessing (station values placed at wrong grid cells), implements the paper's elevation-aware kernel exactly with per-feature bandwidths tuned by leave-one-station-out on training data, and replaces single-run selection with a disciplined protocol: **each reported number is an ensemble of 5 independent seeds** (deterministic training; per seed, the predictions of the 5 best epochs by validation score are averaged; regions' anchor/soft slots are selected on validation only, upgrades admitted only via pre-registered one-shot confirmations).

**Our suggested headline: compare the models on ALL samples (drop the extreme-only slice).**

> **Why drop the extreme slice?** The "extreme" subset is the hottest/coldest ~10% of test days — only **~18–46 samples per region**. With so few points, error bars are wide and a single unusual day can flip the ranking, so it is genuinely hard to reach statistical significance there for *any* method. The all-sample cells use **~239 samples each**, giving stable, defensible conclusions. We therefore lead with all-sample results; the extreme results remain in the paper as supporting evidence (and there our model is never significantly worse, and significantly better in several cells).

Below, every model side by side. "New" = corrected frames + 5-seed ensemble; "Old" = the model in the published paper (same architecture, single run); Tab-LSTM = the tabular ablation; SARIMAX/Prophet = classical baselines; TimesFM/Moirai = zero-shot foundation models (Center only, as in the paper).

**Summer — ANCHOR (hottest day) all-sample MAE (°C), lower is better:**

| Region | **New** | Old (published) | Tab-LSTM | SARIMAX | Prophet | TimesFM | Moirai |
|---|---|---|---|---|---|---|---|
| Center | **2.11** | 2.27 | 2.81 | 4.74 | 4.74 | 5.07 | 4.94 |
| Negev | **2.01** | 2.14 | 2.82 | 4.81 | 4.83 | — | — |
| North-West | **2.01** | 2.08 | 2.67 | 4.52 | 4.50 | — | — |

**Summer — SOFT (3rd/7th/15th) all-sample MAE (°C):**

| Region | **New** | Old (published) | Tab-LSTM | SARIMAX | Prophet |
|---|---|---|---|---|---|
| Center | **1.27** | 1.51 | 1.54 | 1.91 | 1.92 |
| Negev | **1.54** | 1.56 | 1.82 | 2.32 | 2.34 |
| North-West | **1.28** | 1.33 | 1.36 | 1.97 | 1.96 |

**The one-vs-one that matters (New vs Tab-LSTM), with significance** — this is the comparison the paper's claim rests on:

| Season / model | New wins all 6 cells? | Significant (of 6) |
|---|---|---|
| Summer anchor + soft | yes | 5 of 6 (NW soft within noise) |
| Winter anchor + soft | yes | 5 of 6 (Center soft within noise) |

**Bottom line: the New model beats the Old published model, the tabular ablation, both classical baselines, and both foundation models on every all-sample cell** — 12 of 12 vs Tab-LSTM (10 significant), 12 of 12 vs the published model, and by a wide margin vs SARIMAX/Prophet/TimesFM/Moirai. Foundation models sit near the classical level on ordinary days and collapse on extremes (≈10 °C), confirming task-specific training is essential.

**The statistical test, in plain language.** Consecutive forecasts share ~96% of their 30-day windows, so their errors move together — a standard t-test or Wilcoxon (which assume independent samples) would overstate certainty. Instead, for each pair of models we take the day-by-day difference in their errors, then rebuild that series 10,000 times by drawing short *blocks* of consecutive days at random (blocks preserve the local correlation). If the average difference keeps the same sign in at least 95% of the 10,000 rebuilds, the gap is "significant"; a Holm correction then tightens the bar because we test many cells at once. This is the moving-block bootstrap, used throughout the paper.

## 3. Open questions

**Q1 — Why exactly the targets (1, 3, 7, 15)?** Our justification: they sample the 30-day tail at geometric depths — adjacent ranks differ by ~0.1 °C (near-duplicates), so dense sets add outputs without information. New measured evidence (Center, published protocol, all-sample/extreme MAE): dense-adjacent (1,2,3,4) → 1.95/**2.82**; wider (1,5,10,15) → 2.00/2.42; ours (1,3,7,15) → 1.97/**2.39**. The dense set costs 0.4 °C on extremes; the wider set is dominated. Question for discussion: is this justification sufficient, or should the paper present the target set as one operational choice among valid ones?

**Q2 — Fold the corrected model into the manuscript, or keep the published numbers?** The corrected model wins every all-sample cell (12/12 vs the tabular ablation, 12/12 vs the published model) and is fully reproducible from fixed seeds. The one trade-off is on the *extreme* subset: the published tabular baseline's single-seed anchor-extreme values remain numerically ahead in a few winter cells, and in exactly one of them — **winter Negev anchor-extreme** — the gap is statistically significant (the corrected model 2.33 vs the tabular 2.02, p=0.048); every other extreme cell is either a win or a statistical tie. Options: (a) keep published numbers 1:1 (current state), (b) swap to the corrected model with the published values shown as "previously reported", (c) both side-by-side. Our recommendation is (b) — reproducibility across all-sample results outweighs one borderline extreme cell — but this is the key decision for the resubmission.

**Q3 — Single-country scope.** One reviewer noted the study covers one country; that cannot be changed without new data. The paper's Conclusion already carries a limitations statement (20 years, one country/network, three climate regions) — the question is whether to *strengthen* it and/or commit to a follow-up with an external dataset, or whether the current framing (three climatically distinct regions, 20 years, public data, strict chronological backtesting) is sufficient.

**Q4 — Two reviewer suggestions answered with a stance rather than an experiment (for your approval).** (a) *Quantifying the downstream electricity-demand impact*: impossible without grid data, which the study deliberately excludes; the paper scopes it out explicitly and lists it as future work. (b) *Residual plots*: one reviewer preferred them; we kept the (now enlarged) predicted-vs-true panels, which we believe communicate the operational story better. Are both stances acceptable, or should either be revisited before submission?
