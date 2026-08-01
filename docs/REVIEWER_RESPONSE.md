# Reviewer-response coverage matrix (every point → response → status)

Status key: ✅ done · 🔧 in progress · ⏳ planned · 🔒 constraint (3 clusters, 1 country — not changing) · 🚫 out-of-scope
Pillars: P1 reproducibility · P2 baselines · P3 ablations/justification · S2 presentation

**SCOPE CHANGE (per author):** the paper is being **reframed to extreme-temperature
forecasting** — the electricity/peak-demand application is being dropped. DO NOT edit
the paper's electricity sections (author handles the reframe) and DO NOT build further
electricity/downstream analysis. Concerns that were premised on the electricity claim
(downstream demand/reserve quantification, deployment plans) become **out-of-scope**:
they are addressed by *removing the over-claim*, not by satisfying it. The
`downstream_proxy.py` artifact is PARKED (not part of the new direction).

## Reviewer 1 (weak accept)
| # | concern | response | status | artifact |
|---|---|---|---|---|
| R1.1 | one country/network/3 regions (generality) | scope is deliberate; frame as case study + state in limitations; method is general | 🔒 + S2 | paper limitations |
| R1.2 | downstream demand/reserve not tested | OUT OF SCOPE: paper reframed to extreme-temperature; electricity claim dropped | 🚫 | n/a (downstream_proxy parked) |
| R1.3 | 1/3/7/15, 180-window, interp bandwidth, loss weights, extreme weighting not ablated | full ablation suite | ✅ P3 | A2 window(180 best), exp05 k(=1), A4 loss α/β, A3 kernel, A1 targets ⏳ |

## Reviewer 2 (weak accept)
| # | concern | response | status | artifact |
|---|---|---|---|---|
| R2.1 | dataset size / #samples unclear | full dataset summary table | ✅ P1 | results/dataset_stats.md (813/143/238 etc.) |
| R2.2 | why p1/p3/p7/p15 (why not p2) | geometric tail summary justification + ablate alt sets | ✅+⏳ | BASELINE_PROTOCOL.md; A1 ⏳ |
| R2.3 | public code/GitHub + hyperparameters | clean repo + README + hyperparameter table | 🔧 P1 | repo/ + README (this task) |
| R2.4 | which regions easier/harder | per-region results + difficulty discussion | ✅ data | Negev 2.20 < NW 2.43 (per-region) |
| R2.5 | Fig 3 too small | vector figures | ✅/⏳ S2 | results/figures/*.pdf |

## Reviewer 3 (weak reject)
| # | concern | response | status | artifact |
|---|---|---|---|---|
| R3.1 | not reproducible; base models not cited; LSTM hidden size; no code | repo + README + hyperparam table + cite ConvNeXt/LSTM/TFT/PatchTST | 🔧 P1 | repo/ + paper citations ⏳ |
| R3.2 | hard structure (Sec 3-4); figure quality | restructure; vector figs | ⏳ S2 | paper edits |
| R3.3 | eval brief; extreme-forecast vs full TS (SARIMAX) | we DO derive extremes from SARIMAX forecast (apples-to-apples); practical-impact claim narrowed to extreme-temp | ✅ | BASELINE_PROTOCOL.md |
| R3.4 | individual concept impact (ablation) | full ablation suite | ✅ P3 | exp02-08 |
| R3.5 | Mean-MAE (mixing extreme+rest) unclear | report per-target + all-vs-extreme separately | ✅ | results/summer_per_target.csv (A6) |
| R3.6 | anchor vs soft practical use | explain in framing | ⏳ S2 | paper |
| R3.7 | downstream energy demand quantify | OUT OF SCOPE: electricity claim dropped (reframe) | 🚫 | n/a (parked) |
| R3.8 | feature selection split 3.1/4.4; repetition | restructure | ⏳ S2 | paper |
| R3.9 | baselines forecast→extremes + significance described algorithmically | documented step-by-step | ✅ | BASELINE_PROTOCOL.md + eval_lib significance |
| R3.10 | long lines / formatting | fix | ⏳ S2 | paper |

## Reviewer 4 (reject — the hardest)
| # | concern | response | status | artifact |
|---|---|---|---|---|
| R4.1 | SARIMAX/Prophet weak/controversial → no proof | add strong baselines; keep classical as references | 🔧 P2 | TimesFM ✅ (beaten); Moirai/tab-transformer/recursive ⏳ |
| R4.2 | no recursive/transformer/foundational models | TimesFM 2.5 done (we win on extremes); Moirai + tabular PatchTST/iTransformer + recursive LSTM | 🔧 P2 | results/timesfm_*.csv |
| R4.3 | averaging metrics; individual 3/7/15 not provided | per-target table | ✅ | results/summer_per_target.csv |
| R4.4 | Fig 1 unreadable | vector redraw | ⏳ S2 | paper |
| R4.5 | Fig 2 interpolation not elaborated / how used | elaborate kernel + ablation | ✅ doc | INTERPOLATION_FINDING.md + A3 |
| R4.6 | region characteristics/universality | per-region + discussion | ✅ data/⏳ | per-region results |
| R4.7 | Fig 2 (a)≠(b) locations | fix figure | ⏳ S2 | paper |
| R4.8 | dataset summary (locations/period/#points/#features) | summary table | ✅ P1 | results/dataset_stats.md |
| R4.9 | validation procedure/split/#samples | documented | ✅ P1 | dataset_stats + REGISTRY protocol |
| R4.10 | no algorithm/flowchart | add algorithm box | ⏳ S2 | paper |
| R4.11 | code/datasets/repro | repo + README | 🔧 P1 | repo/ + README |
| R4.12 | Fig 3 thick overlapping lines → residuals | cleaner plots / residuals | ✅/⏳ | results/figures/ |

## Meta-review (decision drivers)
| pillar | response | status |
|---|---|---|
| reproducibility (size/splits/arch/hyperparams/training) | repo + dataset_stats + hyperparam table + README | 🔧 P1 |
| weak baselines (transformer/foundational) | TimesFM ✅; Moirai/tab-transformer/recursive ⏳ | 🔧 P2 |
| design choices not ablated (downstream now out-of-scope) | ablation suite ✅ | ✅ |

## Remaining gaps to close (prioritized)
1. **P2 baselines**: Moirai-MoE + tabular PatchTST/iTransformer (non-spatial) + recursive LSTM. (highest — R4 driver)
2. **P1 repo**: README + requirements + hyperparameter table + run-to-replicate (this task).
3. ~~S1 downstream~~ — OUT OF SCOPE (extreme-temperature reframe).
4. **A1 targets** ablation ({1,3,7,15} vs alternatives).
5. **S2 paper writing**: restructure Sec 3-4; vector Fig 1/2/3 + fix Fig2(a/b); algorithm box; cite base models; formatting; per-region difficulty + anchor/soft framing.
