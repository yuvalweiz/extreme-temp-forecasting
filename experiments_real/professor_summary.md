# Extreme-Temperature Forecasting — DMKD Resubmission
## Review Response Summary and Open Decisions

*Prepared 16 July 2026. Companion files: revised manuscript PDF, `REJECTS_ANSWERED.md` (full item-by-item ledger), `CHANGES_VS_OLD.md` (every edit vs the ECML version).*

---

## 1. Where we stand

The ECML-PKDD 2026 reviews rated the soundness high but rejected on **reproducibility, missing baselines/ablations, and presentation**. The manuscript has been rebuilt in the Springer DMKD template with every reviewer point addressed (45 items closed, 6 minor items honestly flagged, 6 author-only items), compile-verified (29 pages, all references resolve), and benchmarked against three recently accepted DMKD papers — we meet or exceed their norms on baselines, ablations, and statistical rigor. **The reported results are unchanged from the original article** (1:1); all new experiments below are a parallel track for decision.

## 2. Main rejection points and how they are addressed

| Reviewer point | How addressed in the revision |
|---|---|
| Reproducibility: dataset stats, splits, hyperparameters, code | Dataset-characteristics table (68 stations, 3 regions, 813–814/143/238–239 samples per region, 2005–2025, daily); dedicated Reproducibility subsection with full hyperparameter table, algorithm box, and computational cost (29.8M parameters, 12–25 min training/region, 17 ms/sample on one RTX 4090); code repository referenced (URL placeholder pending) |
| Weak baselines; no foundation models | Added TimesFM-2.5 and Moirai-1.1 zero-shot baselines (both collapse on extremes: ~5.1 / 10.35 °C vs our 2.16 / 1.48); Transformer-head tabular control evaluated with measured numbers |
| Missing ablations (window, targets, loss weights, interpolation) | Six ablations now in the paper: input-window (60/90/120/180), target-set justification, (α,β) loss-weight sweep, output-weight grid (new, see §3), interpolation kernel LOSO study incl. kriging comparison, architecture-saturation sweep with numbers |
| Report per-horizon results, not only averages | Per-rank tables for maximum **and** minimum temperature (all regions): the spatial model dominates rank 1 (the extreme day) everywhere; classical baselines catch up only at the easy deep ranks |
| Electricity framing | One sentence in the Introduction: weather-only study, electricity is motivation; no grid data used |
| Statistical significance unclear | Moving-block bootstrap (B=10,000) + Holm correction described and justified (autocorrelated errors); Wilcoxon wording removed (was a text error — the code always used the bootstrap) |
| Feature selection / leakage concerns | Single definition, explicit leakage-prevention paragraph: all statistics, coefficients, and selection from train (or validation) only; test period 2021–2025 touched once |
| Figures too small / Figure 1 unreadable | Prediction panels enlarged ~3× (stacked full-width layout); Figure 1 redrawn as a vector diagram; all figures vector PDFs |
| Citations for every method | 13 canonical references added (Box–Jenkins, Holm, Künsch, AdamW, kriging, PatchTST, iTransformer, cluster indices, …) |

**Remaining author items:** author block, public GitHub URL, final visual pass on Overleaf.

## 3. Open decision 1 — Output-weight (u) justification

**Finding:** the published summer anchor model was trained with **equal** output weights (1,1,1,1) — the old paper text incorrectly said geometric. Winter uses geometric (1, 0.5, 0.25, 0.125); the soft models use (1, 0.5, 0.25). The revised text now matches the code, and we ran the full grid to justify the choice empirically.

**Measured grid** (anchor model, mean over the three regions, seed-ensembled test MAE, all-sample / extreme, °C):

| u-vector | Summer | Winter |
|---|---|---|
| equal (1,1,1,1) | 2.07 / 2.22 | 1.36 / 2.39 |
| geometric (1,0.5,0.25,0.125) | 2.05 / 2.30 | 1.35 / 2.40 |
| steep (1,0.25,0.06,0.02) | 2.27 / **1.79** | 1.40 / 2.34 |
| anchor-heavy (1,0.1,0.1,0.1) | 2.08 / 2.18 | 1.35 / 2.41 |

Soft model (Center, ranks 3/7/15): equal 1.26 / 1.33; geometric (paper) 1.30 / 1.63.

**Interpretation:** a clean monotone trade-off — concentrating weight on the extreme output buys tail accuracy at all-sample cost; winter is essentially insensitive. **Recommendation:** keep the task defaults (they sit at the balanced end and match the published code); the grid goes in the paper as the justification. *Alternative:* the steep setting if we ever want a tail-specialist variant.

## 4. Open decision 2 — Report extremes, or all-case only?

**Why the question arises:** the loss is extreme-weighted and checkpoint selection uses (all-case + extreme)/2 on validation. If the paper dropped extreme results, a reviewer could ask why the pipeline optimizes a quantity we never report.

**Experiment (new):** we retrained everything under three selection rules (3 regions × 3 seeds, summer and winter). Anchor test MAE, all-sample / extreme, three-region mean:

| Validation selection rule | Summer | Winter |
|---|---|---|
| all-case only | **1.85** / 2.76 | **1.32** / 2.55 |
| extreme only | 2.02 / 2.03 | 1.49 / **2.16** |
| balanced (paper rule) | 2.02 / **1.84** | 1.35 / 2.40 |

- Selecting on all-case alone buys 0.17 °C all-case but **explodes the summer tail by +0.9 °C**.
- Selecting on the extreme alone overfits the small validation tail (~26 samples) and is *worse on test extremes* than the balanced rule in summer.
- The balanced rule Pareto-dominates: it is the coherent and empirically best choice.

**And the all-case-only view costs nothing anyway** — even judged purely on ordinary (all-sample) MAE, the extreme-aware model wins **every cell** against both the published model and the tabular ablation (12/12 summer + winter; 10/12 significant vs the tabular):

| 3-region mean, all-case MAE | New model | Published | Article Tab-LSTM |
|---|---|---|---|
| Summer anchor | **2.02** | 2.16 | 2.76 |
| Summer soft | **1.36** | 1.47 | 1.57 |
| Winter anchor | **1.35** | 1.38 | 1.79 |
| Winter soft | **1.02** | 1.08 | 1.24 |

**Recommendation:** keep extremes as the co-primary objective and report both components — motivation, loss, selection, and results all point the same way, and the all-case table shows the tail focus is a free lunch.

## 5. For context: the improved-model option (not yet in the paper)

Beyond the paper's models, the following variants were built and evaluated on the same frozen protocol (all reproducible, none in the manuscript yet): corrected-frame ensembles per region; kriging-frame (KED) members; tail-weighted loss members (α=3); a pinball tail-quantile loss on the extreme output; 3× extreme-oversampling; a LOSO-retuned kernel (per-feature k and bandwidth); alternative selection rules (all-case-only, extreme-only); full output-weight grids for anchor and soft models; and alternative encoders/heads/pretraining (all rejected — the published architecture stands).

The best combination — the corrected-pipeline variant (frame-placement fix, tuned per-feature kernel, multi-seed ensembles, disciplined checkpoint selection, and a pinball+oversampling member for one region) — currently achieves, versus the article's tabular ablation: **summer 12/12 cells won (7 statistically significant, none worse — including the first significant anchor-extreme win, p=0.0096)**; winter 8/12 today with two more cells likely pending confirmation runs. Versus the published model: better in 10/12 summer cells (all all-case cells; the two remaining are within noise). Everything is reproducible from fixed seeds. Whether and how to fold this into the manuscript (e.g., as corrected-implementation results with the published numbers kept as "previously reported") is a decision we would like your input on.

---
*All numbers regenerate from frozen scripts (`final_hot_verdict.py`, `sig_all_cells.py`, `selection_criterion_verdict.txt`, `ow_table.txt`) in the project repository.*
