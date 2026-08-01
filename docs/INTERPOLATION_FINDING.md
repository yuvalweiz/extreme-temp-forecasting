# Finding: interpolation code does not match the paper (the "IDW bug")

**Status: confirmed.** The original preprocess notebook (cell 13) does *not*
implement the interpolation written in the paper (paper.tex eq. d3d / exp_weights).
This is the bug to fix, and it doubles as ablation **A3** (interpolation /
bandwidth / elevation) that reviewers R1 and R4 explicitly asked for.

## The discrepancy

| component | paper (eq. d3d) | code (cell 13) |
|---|---|---|
| horizontal term | `d_horiz / d_horiz,max` (max-normalized → [0,1]) | raw km |
| elevation term | `k_f · |d_elev| / d_elev,max` (max-normalized) | `0.001 · k_f · Δh_m` (raw metres, magic 0.001) |
| 3D distance | `√(h_norm² + (k_f·e_norm)²)` | `√(d_km² + (0.001·k_f·Δh)²)` |
| kernel | `exp(−γ·d_3D)`, γ tunable bandwidth | `exp(−1.0·d_eff)` over raw km (≈ nearest-neighbour) |

Because the code never max-normalizes, the horizontal (tens of km) and elevation
(`0.001·k_f·Δh`, i.e. ~0.2–2 km for temperature `k_f∈[2.3,20.9]`) terms are on
wildly different scales, so **elevation is essentially inert for the temperature
channels**, and `exp(−d_km)` over raw km collapses to the nearest station. The
paper's normalized `d_3D` with a tuned γ is smoother and genuinely
elevation-aware — which is the contribution reviewers *praised*.

## Concrete impact (self-test in `src/data/interpolation.py`)

High pixel (600 m) near an 800 m hill station, `k_f=2.3`:

```
as-coded  w=[0.792 0 0 0.208]  -> T = 29.79 °C   (ignores the hill; hot lowland value)
paper     w=[0.006 0.981 0 0.011] -> T = 22.15 °C (pulled to the cool hill station)
```

A **7.6 °C** difference at a single pixel. The as-coded frames are not the
elevation-aware representation the paper claims.

## k-value derivation — verified correct

`k_values.csv` column `k` = `|StdEffect(height) / StdEffect(distance)|` from a
per-feature station-pair regression (e.g. prs_stn 54.26/0.228 ≈ 237.6 ✓), exactly
the paper's dimensionless ratio (eq. ~122). Temperature channels: `k_f` ∈
[2.32 (max_dry_temp), 20.93 (tmp_air_wet)]. Only the *use* of `k_f` in the kernel
diverges, not its derivation.

## Decision / plan

Implement the paper-accurate kernel (`weights_paper` in `interpolation.py`) and
**rebuild frames for the A3 ablation**, comparing on identical splits:

1. `as-coded` (current frames — already trained; MAE_hot avg 1.82).
2. `paper γ*` (max-normalized, k_f on elevation; tune γ on validation only).
3. `paper, no-elevation` (drop the elevation term) — isolates the elevation contribution.

Expected story for the resubmission: either the fix **improves** results (then we
report the corrected, genuinely elevation-aware method as SOTA), or it is
**neutral** (then we report it as a robustness/ablation and keep the as-coded
variant, with the paper text corrected to match the code). Either way the
paper↔code consistency problem and the R1/R4 ablation request are resolved.

γ is the single tunable bandwidth; the A3 sweep over γ (and elev on/off) is the
justification reviewers asked for. No leakage: γ selected on validation only;
normalization maxima are computed per frame from geometry (not from targets).
