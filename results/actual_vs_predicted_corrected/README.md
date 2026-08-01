# Row-Level Predictions Behind Every Reported Number (final manuscript)

This directory contains the complete per-sample actual-vs-predicted data for every model,
region, season, and target reported in the paper's result tables, plus the final metrics
and significance files. Every table cell in the manuscript recomputes exactly from these
files; nothing is hand-edited.

Layout and column dictionary are identical to `../..` conventions:
`{summer,winter}/{Center,Negev,Northwest}/<Model>__<anchor|soft>.csv` with columns
`sample_id, date, region, season, target, model, split, actual, predicted, residual,
abs_error, sq_error` (+ per-rank `actual_p*`/`predicted_p*` in soft files).

Models: `ConvNeXtTiny-TFT` (the paper's proposed model: per-region/target
validation-selected, seed-ensembled configurations), `ConvNeXtTiny-LSTM` (LSTM-head
variant, 5 seeds), `Tab-LSTM` (the spatial-structure ablation: per-station 612-dim
input, LSTM head, 5 seeds). Classical (SARIMAX, Prophet) and foundation
(TimesFM-2.5, Moirai-1.1) rows of the tables recompute from
`../../data/published_predictions/` and `../timesfm_*.csv` / `../moirai_*.csv`
respectively, as in the base evaluation scripts.

Metric definitions (per the manuscript's Evaluation Metrics section):
* MAE = mean of `abs_error`; RMSE = sqrt(mean of `sq_error`) — full test set, no subsets.
* Soft files implement MAE₃/RMSE₃ (per-rank errors averaged over ranks p3/p7/p15).
* UPE (under-prediction error) = mean(max(0, actual − predicted)) for the
  maximum-temperature task and mean(max(0, predicted − actual)) for the
  minimum-temperature task, i.e. mean(max(0, σ·residual·(−1))) with σ as defined in the
  paper — computable directly from the `residual` column.

`metrics_summary_corrected.csv` holds every final MAE/RMSE (+ 3-region means);
`significance_corrected.csv` holds the moving-block-bootstrap results
(B=10,000, block length 4, two-sided, Holm-corrected) behind the significance table and
the ablation section's p-values.
