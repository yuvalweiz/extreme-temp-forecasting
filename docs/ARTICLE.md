# ARTICLE.md — the paper, big picture

Source: `latech/paper.tex` (original ECML submission) ≡ `dami_submission/main.tex` (DAMI, body
byte-identical). Reframe target: extreme-temperature (electricity-demand framing softened).

## One-line thesis
An operator-oriented **spatiotemporal** framework forecasts, a **month ahead**, the regional
**temperature extremes** of the next 30 days — from past weather-station observations turned into
**elevation-aware station→grid frames** → **ConvNeXt-Tiny** (per frame) → **temporal head
(TemporalFusionLite / LSTM)** → **extreme-aware loss**. Claim: the **spatial** representation adds
signal **beyond region-aggregated tabular histories**, especially for the hot extreme.

## Scope (what the paper actually delivers) — the big picture I must not lose
- **Three Israeli climate regions**: Center, North-West, Negev. Metrics are **AGGREGATED across
  the 3 regions** in the main tables (not per-region).
- **Two seasons × two target types = the model family:**
  - **PRIMARY / "anchor" targets**: the **hottest day** (summer, daily max) AND the **coldest day**
    (winter, daily min) of the next 30 days. THESE ARE CO-PRIMARY — both matter equally.
  - **SECONDARY / "soft" targets**: the **3rd / 7th / 15th** hottest (and coldest) day — a
    less-extreme "risk profile". Explicitly secondary.
- **Order-statistic targets**: for a 30-day window, pick sorted-day ranks [1,3,7,15] (hot: desc on
  max; cold: asc on min). Anchor = rank 1; soft = ranks 3/7/15.

## The headline METRIC (critical — I was using the wrong one)
- Errors in °C. **Extreme = top/bottom 10%** of test samples (p90 for hot, p10 for cold) — NOT p95.
- Three error columns: **MAE_all** (all samples), **MAE_ext** (the top/bottom-10% subset),
  **Mean MAE = (MAE_all + MAE_ext)/2** ← **the main ranking criterion**. RMSE variants for completeness.
- So "beating the paper" = beating its **Mean MAE** (3-region aggregate), on the anchor targets,
  with the extreme subset defined at 10%. Per-region p1 all-case MAE (what I first reported) is
  related but NOT the headline number.

## Published results to beat (3-region aggregate, TFT)
| target | MAE_all | MAE_ext(10%) | **Mean MAE** |
|---|---|---|---|
| Hottest day (anchor) | 2.160 | 1.483 | **1.822** |
| Coldest day (anchor) | 1.385 | 2.096 | **1.740** |
| Max soft (3/7/15 mean) | 1.322 | 1.212 | 1.267 (TFT best) |
| Min soft (3/7/15 mean) | 1.033 | 1.359 | 1.196 (LSTM best) |
Baselines beaten: Prophet, SARIMAX (both ~9°C ext on hot), **Tab-LSTM (the tabular ablation —
tabular input instead of spatiotemporal frames)**. Foundational (TimesFM/Moirai) added in rebuttal.

## Significance (moving-block bootstrap, per region, ~239 samples, 2000 resamples)
- Hot anchor: TFT significant **3/3** regions vs ALL baselines. Strongest, most consistent result.
- Cold anchor: significant vs Tab-LSTM 3/3; vs classical 2/3 (NW not).
- Soft: weaker significance (esp. min). Consistent with the small soft-target gaps.

## Structure (sections)
Intro → Related work (demand planning; forecasting baselines; spatiotemporal DL; extreme events)
→ Methods (spatial interpolation; region construction; ConvNeXt+head pipeline; extreme-aware loss;
extreme definition & metrics) → Experimental setup (IMS dataset/regions/targets; chronological
backtesting; normalization; features; splits/protocol) → Results (anchor primary; soft secondary;
significance) → Conclusion (+ GenAI declaration).

## Reviewer rejects → the article's job (see REVIEWER_EVIDENCE.md)
P1 reproducibility (dataset stats/splits/hyperparams/public code) · P2 baselines (recursive +
transformer + FOUNDATIONAL — added) · P3 ablations (targets, window, interpolation bandwidth, loss
weights — see ABLATIONS.md). Plus figures, per-target metrics, regional difficulty.

## DAMI article plan (author decision 2026-07-02)
For NOW the DAMI submission is **1:1 with the original + DAMI Springer template + an Ablations
section**. Headline numbers stay the original's; the tabular ablation stays the key spatial control.
**Improvements are DEFERRED** and folded in only once confirmed to hold across the full picture
(hot + cold anchors, all regions). See below.

## Our results vs the paper (status, in the paper's frame) — DEFERRED for the article
- **Hot anchor**: our recipe (V2 + cosine/EMA + in-domain pretrain + [1,5,10,15]) beats the
  published per-region on the hottest-day point (Center 1.997<2.265 etc.). Must be RE-EXPRESSED in
  the paper's Mean MAE (all+ext@10%)/2, 3-region aggregate, to claim a headline improvement.
- **Cold anchor** (co-primary): cold pipeline now reproducible (`COLD=1`); improved-recipe cold
  rollout not yet confirmed.
- **Soft (secondary)**: our model ≈ TIED with the published soft (no clear win; hot-focused
  improvements don't transfer). Fine — soft is secondary.
- **Spatial-vs-tabular (the core claim)**: fair 68-station tabular control ties spatial on
  all-sample but spatial wins the EXTREME tail decisively (1.541 vs 2.213) → supports the thesis in
  the paper's own Mean-MAE terms. See ABLATIONS.md §E.

## Open TODOs for the write-up
Re-express our gains in Mean MAE @10% (3-region aggregate) for a like-for-like claim; fill window
ablation; decide EXP-vs-IDW ablation; author/affiliation block + fix abstract "Wilcoxon"→bootstrap;
vector figures. See [[PIPELINE.md]], [[CODE.md]], [[DATA.md]], ABLATIONS.md, DAMI_ARTICLE_GUIDE.md.
