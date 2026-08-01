"""
Canonical pipeline config — ported 1:1 from the author's run_grid.py (the script that
produced the PUBLISHED ConvNeXt-TFT/LSTM models in
`Deep Learning Models/Cluster <Region>/grid_ckpts_FINAL*`).

Nothing here is a redesign. Values match the published runs exactly:
  - extreme threshold = TRAIN p95 (EXTREME_Q=0.95)   [loss masking + selection]
  - out_weights = (1,1,1,1)  (equal weight over the 4 order-stat outputs)
  - loss = WeightedMAE_ExtremeUnderOnly + ExtremeUnderHinge  (NO percentile w(r), NO k)
  - arch fixed: d_model=256, nhead=4, layers=2
  - AdamW (lr_bb=1e-5, lr_head=1e-4, wd=1e-4), grad_clip=1.0, AMP, NO scheduler, NO EMA
  - patience=30, epochs=120, batch=4, seed=333, DOM in {1,7,14,21,28}
  - selection = avg(val_mae_all, val_mae_ext, underrate_ext_all_outputs)
Published per-region anchor configs (val-selected): Center a2-b1, Negev a3-b1.5, NW a1-b0.
"""
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
from repo_paths import canonical_dataset  # noqa: E402

# Default = the canonical frame datasets under DATA_ROOT (author layout). Override the
# location with DATA_ROOT=... or point a single run elsewhere with DATASET_DIR=...
# (e.g. the bundled data/stationvec_<Region> for STATIONVEC/PAPERW runs).
REGION_DATASET = {r: canonical_dataset(r, check=False)
                  for r in ("Center", "Negev", "Northwest")}

# COLD (winter / MIN targets) datasets — same frame structure as hot, cold order-stat
# targets (output 0 = coldest day). Used only when COLD=1 (see train.py). Faithful to
# run_grid_cold_winter.py. Cold recipe DIFFERS from hot:
#   - out_weights = (1.0, 0.5, 0.25, 0.125)  [GEOMETRIC, not equal (1,1,1,1)]
#   - y_transform = asinh, wd_backbone = 1e-3 (hot uses 1e-4)
#   - extreme = LOW tail, EXTREME_Q = 0.10 (train p10); selection/eval only, not in loss
#   - loss = PercentileWeightedMAE_Min (+ optional PercentileWeightedHinge_Min)
#   - cosine LR (eta_min=1e-6) + EMA-of-val-score smoothing (alpha=0.35), always on for cold
#   - norm file: norm_stats_extremes_full_MIN.npz ; splits: split_{train,val,test}.csv (no tag)
COLD_OUT_WEIGHTS = (1.0, 0.5, 0.25, 0.125)   # geometric (cold only)
COLD_EMA_ALPHA_SCORE = 0.35                  # EMA smoothing of the val selection score
REGION_DATASET_MIN = {r: canonical_dataset(r, cold=True, check=False)
                      for r in ("Center", "Negev", "Northwest")}
# Each region's PUBLISHED anchor config (head_type, under_alpha_ext, beta_hot, dropout)
PUBLISHED = {
    "Center":    dict(head_type="temporalfusion", alpha=2.0, beta_hot=1.0, dropout=0.1),
    "Negev":     dict(head_type="temporalfusion", alpha=3.0, beta_hot=1.5, dropout=0.1),
    "Northwest": dict(head_type="temporalfusion", alpha=1.0, beta_hot=0.0, dropout=0.1),
}


@dataclass
class Config:
    # data
    dataset_dir: str = REGION_DATASET["Center"]
    tag_norm: str = "yNONE_v1"                 # norm/splits file tag
    keep_dom: tuple = (1, 7, 14, 21, 28)
    extreme_q: float = 0.95                    # TRAIN quantile defining "extreme" (p95)
    batch_size: int = 4
    num_workers: int = 4
    eps: float = 1e-8

    # model (FIXED in every published run)
    head_type: str = "temporalfusion"          # temporalfusion | lstm | patchtst | itransformer
    d_model: int = 256
    nhead: int = 4
    layers: int = 2
    dropout: float = 0.1
    backbone_chunk: int = 256
    pretrained: bool = True
    use_out_affine: bool = True
    out_dim: int = 4
    out_weights: tuple = (1.0, 1.0, 1.0, 1.0)
    backbone_pool: str = "avg"                  # avg(published) | max | avgmax | attn
    backbone_name: str = "convnext_tiny"        # convnext_tiny(published) | convnextv2_tiny | convnextv2_nano ...
    head_layers: int = None                     # None = published derivation (TFT=2); int = explicit
    no_wd_norm: bool = False                     # True = exclude LayerNorm/bias from weight decay
    spatial_stage: int = None                    # None=GAP(published); 0/1/2 = earlier stage map (A1: preserve spatial tokens)

    # loss
    under_alpha_ext: float = 2.0               # under-prediction multiplier ON EXTREMES only
    beta_hot: float = 1.0                       # extreme under-hinge weight
    use_extra_hinge: bool = True

    # optim / schedule
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    epochs: int = 120
    patience: int = 30
    use_amp: bool = True
    seed: int = 333

    # io
    ckpt_root: str = "./pipeline_runs"
    run_tag: str = "RUN"

    @classmethod
    def for_region(cls, region: str, **overrides):
        pub = PUBLISHED[region]
        c = cls(dataset_dir=REGION_DATASET[region], head_type=pub["head_type"],
                under_alpha_ext=pub["alpha"], beta_hot=pub["beta_hot"], dropout=pub["dropout"])
        for k, v in overrides.items():
            setattr(c, k, v)
        return c
