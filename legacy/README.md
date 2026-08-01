# legacy/ — superseded code (provenance only)

**Superseded early reimplementation — NOT used for any reported result.**
The canonical pipeline is [`src/pipeline/`](../src/pipeline) (a faithful port of
the author's `run_grid.py`). This directory is kept only for provenance.

Contents:

- `train_reimpl/` — an earlier reimplementation (`train_grid_hot.py`,
  `train_from_frames.py`, `models.py`) that produced **invalid early results**.
  It is not the canonical pipeline and should not be used.
- `aggregate_experiments.py` — aggregation script tied to the old
  `article /experiments/exp0*` layout (reads `train_grid_hot` /
  `train_from_frames` `meta.json`). Does not apply to the canonical pipeline's
  output layout.

Do not use anything here to reproduce or extend results. See the top-level
[`README.md`](../README.md) for the canonical workflow.
