# Architecture & training hyperparameters (answers R2.3 / R3.1)

All values are the defaults in `src/train/models.py` + `src/train/train_grid_hot.py`.
Base models cited: ConvNeXt (Liu et al. 2022), LSTM (Hochreiter & Schmidhuber 1997),
Temporal Fusion Transformer (Lim et al. 2021), PatchTST (Nie et al. 2023),
iTransformer (Liu et al. 2024).

## Input
| item | value |
|---|---|
| frame grid | 44 × 137 (study-wide, elevation-aware) |
| channels (C) | 9 (dry/wet/dew temps + min/max heat-stress; pressure excluded) |
| input window (T) | 180 days |
| forecast window | 30 days |
| targets | p1/p3/p7/p15 = 1st/3rd/7th/15th hottest(coldest) day |
| x normalization | per-channel mean/std, **train only** |
| y normalization | robust median/IQR, **train only** |

## Spatial encoder
| item | value |
|---|---|
| backbone | ConvNeXt-Tiny (timm `convnext_tiny`, ImageNet-pretrained) |
| in_chans | 9 | 
| pooling | global average → 768-d per-frame embedding |
| per-frame chunking | 256 frames/forward (memory) |

## Temporal heads (d_model = 256, dropout = 0.1 for all)
| head | key params |
|---|---|
| TemporalFusionLite | nhead=4, 2 attention layers, GLU gating, last-step readout |
| LSTM | **hidden size = 256**, 1 layer, unidirectional |
| PatchTST | nhead=4, 2 layers, patch_len=30, stride=30 |
| iTransformer | nhead=4, 2 layers, variate-as-token |
| output | OutputAffine (per-target scale+bias) → 4 (or 3 for soft) outputs |

## Loss (percentile-weighted, extreme-aware)
| item | value |
|---|---|
| base | `w(r)·Σ uⱼ·ℓᵢⱼ`, w(r)=1 if r<0.5 else exp(k·(r−0.5)), r = train-ECDF rank of true p1 |
| underprediction | on hot samples (r≥0.5), errors in the risky direction ×α |
| hinge | + β·relu(true_p1 − pred_p1)·w(r) |
| output weights u | (1, 0.5, 0.25, 0.125) anchor / (1, 0.5, 0.25) soft |
| swept | k∈{1,2,3}, α∈{1,2,3}, β∈{0,1,1.5}; **selected k=1** (ablation exp05) |

## Optimization
| item | value |
|---|---|
| optimizer | AdamW, split param groups |
| lr (backbone / head) | 1e-5 / 1e-4 |
| weight decay (backbone / head) | 1e-3 / 1e-4 |
| schedule | CosineAnnealingLR, eta_min=1e-6 |
| grad clip | 1.0 | mixed precision | AMP (fp16) on CUDA |
| batch size | 4 | epochs | 150 | patience | 30 |
| seed | 333 (+ seed-averaging over 111/222/444) |

## Selection (no leakage)
| item | value |
|---|---|
| metric | EMA(α=0.35) of avg(val_mae_all, val_mae_ext) on p1 |
| extreme threshold | train p90 (hot) / p10 (cold) |
| split | chronological train/val/test (test strictly later) |
| rule | **best config chosen on validation; test read once** |
