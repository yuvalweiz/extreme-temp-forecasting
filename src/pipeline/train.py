"""
Training — run_one_config ported 1:1 from run_grid.py (CELL 3). Optional knobs
(scheduler, EMA) DEFAULT OFF so the base run is byte-faithful to the published model;
they exist only so the optimization scan can toggle them without forking the loop.

CLI:  python -m pipeline.train  (env: REGION, HEAD, ALPHA, BETA, DROPOUT, SEED,
      EXTREME_Q, WINDOW, OUT_WEIGHTS, SCHEDULER, EMA, EPOCHS, PATIENCE, CKPT_ROOT, RUN_TAG;
      PAPERW=<interp_weights .npz> + DATASET_DIR=<stationvec dir> -> synthesized-frame
      interpolation ablation, all other knobs unchanged;
      DENSE_PRETRAIN=<n_epochs> -> TASK B: pretrain the same arch (out_dim=30) on the raw
      next-30-day daily cluster series, transfer all but the final layer, then run normally;
      MULTITASK=1 -> TASK C: one shared backbone + Center/Negev/Northwest heads on the
      per-split tag intersection (X from the REGION/primary pipeline, plain or PAPERW synth);
      MT_LIMIT=<n> caps samples per split (smokes only);
      HYBRID=1 (requires PAPERW) -> hybrid model: per timestep, CNN(synth frame) features
      CONCAT a Linear(D_SV, default 256) projection of the raw normalized station vector
      -> temporal head (strictly more information than the fair tabular control);
      SAVE_TOPK=<k> -> additionally save preds_{test,val}_topk.csv = per-tag MEAN of the
      k best epochs' (by the selection score) predictions + topk_eps/test_topk in meta.json;
      BACKBONE3D=1 -> joint space-time encoder (model.SpatioTemporal3D, factorized (2+1)D
      convs over the whole (T,C,H,W) window) instead of the per-frame ConvNeXt; plain
      frames / PAPERW paths only (asserts against COLD/MULTITASK/HYBRID/TABULAR/
      STATIONVEC/DENSE_PRETRAIN/PRETRAINED);
      best.pt/preds_test.csv/preds_val.csv semantics unchanged)
"""
import os
import sys
import time
import json
import copy
import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))   # <repo>/src
sys.path.insert(0, _SRC)
from repo_paths import experiments_root
from pipeline.config import (Config, REGION_DATASET, PUBLISHED,
                             REGION_DATASET_MIN, COLD_OUT_WEIGHTS, COLD_EMA_ALPHA_SCORE)
from pipeline.data import build_data
from pipeline.model import ConvNeXtTiny_WithHead, HeadOnlyModel
from pipeline.loss import (WeightedMAE_ExtremeUnderOnly_Celsius, ExtremeUnderHinge,
                           PercentileWeightedMAE_Min, PercentileWeightedHinge_Min)
from pipeline.metrics import (collect_pred_table, val_select_score_avg3, val_select_score_paper,
                              suite, compute_cold_metrics, val_select_score_cold, suite_cold,
                              val_select_score_component)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


def run_one_config(cfg, loaders, ns, hot_thr, scheduler=None, ema_decay=None,
                   pretrained_path=None, geo_chans=0, warmup=0, deterministic=False,
                   tabular=False, diag_test=False, verbose=True, init_state=None,
                   hybrid=False, d_sv=256, save_topk=0):
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    in_chans = ns["C"] + int(geo_chans); T = ns["T"]

    if hybrid:    # HYBRID (PAPERW synth): frames + raw station vectors -> concat -> head.
        from pipeline.model import ConvNeXtTiny_Hybrid   # batches are HybridBatch objects
        model = ConvNeXtTiny_Hybrid(                     # (duck-typed X: loops unchanged)
            in_chans=in_chans, sv_dim=int(ns["sv_dim"]), T=T, head_type=cfg.head_type,
            d_sv=int(d_sv), backbone_chunk=cfg.backbone_chunk, pretrained=cfg.pretrained,
            d_model=cfg.d_model, nhead=cfg.nhead, layers=cfg.layers, dropout=cfg.dropout,
            use_out_affine=cfg.use_out_affine, out_dim=cfg.out_dim,
            backbone_pool=cfg.backbone_pool, backbone_name=cfg.backbone_name,
            head_layers=cfg.head_layers, spatial_stage=cfg.spatial_stage,
        ).to(DEVICE)
    elif tabular:   # non-spatial control: series straight to the head, no ConvNeXt
        model = HeadOnlyModel(F=ns["C"], T=T, head_type=cfg.head_type, d_model=cfg.d_model,
                              nhead=cfg.nhead, layers=cfg.layers, dropout=cfg.dropout,
                              out_dim=cfg.out_dim, head_layers=cfg.head_layers).to(DEVICE)
    elif os.environ.get("BACKBONE3D", "0") == "1":   # opt-in joint space-time 3D encoder:
        from pipeline.model import SpatioTemporal3D  # (2+1)D convs over the whole window
        assert not pretrained_path and init_state is None, \
            "BACKBONE3D=1 does not support PRETRAINED / DENSE_PRETRAIN (from-scratch 3D encoder)"
        model = SpatioTemporal3D(
            in_chans=in_chans, T=T, head_type=cfg.head_type, d_model=cfg.d_model,
            nhead=cfg.nhead, layers=cfg.layers, dropout=cfg.dropout,
            use_out_affine=cfg.use_out_affine, out_dim=cfg.out_dim,
            head_layers=cfg.head_layers,
        ).to(DEVICE)
    else:
        model = ConvNeXtTiny_WithHead(
            in_chans=in_chans, T=T, head_type=cfg.head_type, backbone_chunk=cfg.backbone_chunk,
            pretrained=cfg.pretrained, d_model=cfg.d_model, nhead=cfg.nhead, layers=cfg.layers,
            dropout=cfg.dropout, use_out_affine=cfg.use_out_affine, out_dim=cfg.out_dim,
            backbone_pool=cfg.backbone_pool, backbone_name=cfg.backbone_name, head_layers=cfg.head_layers,
            spatial_stage=cfg.spatial_stage,
        ).to(DEVICE)
    if pretrained_path:
        ck = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
        miss, unexp = model.backbone.m.load_state_dict(ck["enc_state_dict"], strict=False)
        print(f"[in-domain pretrain] loaded {pretrained_path} into backbone (missing={len(miss)} unexpected={len(unexp)})")
    if init_state is not None:   # TASK B (DENSE_PRETRAIN): transfer ALL weights except the
        own = model.state_dict()  # final output layer(s) — shape-mismatched (30- vs 4-dim)
        keep = {k: v for k, v in init_state.items() if k in own and own[k].shape == v.shape}
        miss, unexp = model.load_state_dict(keep, strict=False)
        print(f"[dense] transferred {len(keep)}/{len(init_state)} tensors from the dense-pretrained "
              f"model (dropped final-layer shapes; missing={len(miss)} unexpected={len(unexp)})")

    base = WeightedMAE_ExtremeUnderOnly_Celsius(
        y_median=ns["y_median"], y_iqr=ns["y_iqr"], y_transform=ns["y_transform"],
        out_weights=cfg.out_weights, hot_thr=hot_thr, under_alpha_ext=cfg.under_alpha_ext).to(DEVICE)
    loss_fn = (ExtremeUnderHinge(base, hot_thr=hot_thr, beta_hot=cfg.beta_hot).to(DEVICE)
               if cfg.use_extra_hinge else base)
    if os.environ.get("PINBALL"):     # opt-in tail-quantile term on the anchor output
        from pipeline.loss import PinballAnchor
        loss_fn = PinballAnchor(loss_fn, tau=float(os.environ["PINBALL"]),
                                weight=float(os.environ.get("PINBALL_W", "1.0"))).to(DEVICE)

    def _is_no_decay(name, p):
        return cfg.no_wd_norm and (p.ndim <= 1 or name.endswith(".bias"))  # norm weights + biases
    groups = {"bb_d": [], "bb_n": [], "hd_d": [], "hd_n": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        side = "bb" if name.startswith("backbone.") else "hd"
        groups[f"{side}_{'n' if _is_no_decay(name, p) else 'd'}"].append(p)
    opt = torch.optim.AdamW(
        [{"params": groups["bb_d"], "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
         {"params": groups["bb_n"], "lr": cfg.lr_backbone, "weight_decay": 0.0},
         {"params": groups["hd_d"], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
         {"params": groups["hd_n"], "lr": cfg.lr_head, "weight_decay": 0.0}])
    sch = None
    if scheduler == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    ema = copy.deepcopy(model).eval() if ema_decay else None

    best_val, best_ep, best_state, bad = float("inf"), -1, None, 0
    diag_rows = []
    topk = {}    # SAVE_TOPK: {ep: (score, df_val, df_test)} for the current k best epochs
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for X, y_norm, y_raw, tags in loaders["train"]:
            X = X.to(DEVICE, non_blocking=True).float()
            y_raw = y_raw.to(DEVICE, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                loss = loss_fn(model(X).float(), y_raw.float())
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(ema_decay).add_(pm.detach(), alpha=1 - ema_decay)
        if sch is not None:
            sch.step()

        eval_model = ema if ema is not None else model
        df_val = collect_pred_table(eval_model, loaders["val"], ns, DEVICE, cfg.use_amp)
        sel = getattr(cfg, "select", "avg3")
        if sel in ("all", "ext"):   # SELECT ablation: single-component selection
            val_score = val_select_score_component(df_val, getattr(cfg, "select_q", 0.90), sel)
        else:
            val_score = (val_select_score_paper(df_val, getattr(cfg, "select_q", 0.90))
                         if sel == "paper" else val_select_score_avg3(df_val, hot_thr))
        if diag_test:   # val<->test correlation study: store EVERY epoch's predictions
            df_te_ep = collect_pred_table(eval_model, loaders["test"], ns, DEVICE, cfg.use_amp)
            for split, dfx in (("val", df_val), ("test", df_te_ep)):
                d = dfx.drop(columns=["tag_dt"], errors="ignore").copy()
                d.insert(0, "ep", ep); d.insert(1, "split", split)
                diag_rows.append(d)
        if save_topk and ep >= warmup:   # SAVE_TOPK: stash preds when this epoch enters the
            worst = max(topk, key=lambda e: topk[e][0]) if topk else None   # current k best
            if len(topk) < int(save_topk) or val_score < topk[worst][0]:
                df_te_k = (df_te_ep if diag_test else                       # 1 extra test pass
                           collect_pred_table(eval_model, loaders["test"], ns, DEVICE, cfg.use_amp))
                topk[ep] = (float(val_score), df_val, df_te_k)
                if len(topk) > int(save_topk):
                    del topk[max(topk, key=lambda e: topk[e][0])]           # evict the worst
        if os.environ.get("FIXED_STOP", "0") == "1":   # final full-fit: no selection, run to
            best_val, best_ep = val_score, ep           # cfg.epochs exactly, keep the LAST state
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            bad = 0
        elif ep < warmup:                     # min-epoch floor: ignore noisy very-early epochs
            pass
        elif val_score < best_val - 1e-6 or best_state is None:
            best_val, best_ep = val_score, ep
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and (ep == 1 or ep % 5 == 0 or bad == 0 or bad >= cfg.patience):
            print(f"[{cfg.head_type}|a{cfg.under_alpha_ext}b{cfg.beta_hot}|s{cfg.seed}] ep{ep:03d} "
                  f"val_select={val_score:.4f} best={best_val:.4f}@{best_ep} bad={bad}/{cfg.patience}")
        if bad >= cfg.patience:
            break

    if best_state:
        (ema if ema is not None else model).load_state_dict(best_state)
    final = ema if ema is not None else model
    df_val = collect_pred_table(final, loaders["val"], ns, DEVICE, cfg.use_amp)
    df_test = collect_pred_table(final, loaders["test"], ns, DEVICE, cfg.use_amp)
    val_s, test_s = suite(df_val, hot_thr), suite(df_test, hot_thr)

    os.makedirs(cfg.ckpt_root, exist_ok=True)
    run_id = f"{cfg.run_tag}__{cfg.head_type}__a{cfg.under_alpha_ext:g}__b{cfg.beta_hot:g}__s{cfg.seed}"
    run_dir = os.path.join(cfg.ckpt_root, run_id); os.makedirs(run_dir, exist_ok=True)
    if os.environ.get("SAVE_CKPT", "0") == "1" and best_state is not None:   # persist weights
        torch.save({"state_dict": best_state, "best_ep": best_ep, "run_id": run_id},
                   os.path.join(run_dir, "best.pt"))
    df_test.drop(columns=["tag_dt"]).to_csv(os.path.join(run_dir, "preds_test.csv"), index=False)
    df_val.drop(columns=["tag_dt"]).to_csv(os.path.join(run_dir, "preds_val.csv"), index=False)
    meta = dict(run_id=run_id, region_dir=cfg.dataset_dir, head_type=cfg.head_type,
                under_alpha_ext=cfg.under_alpha_ext, beta_hot=cfg.beta_hot, dropout=cfg.dropout,
                extreme_q=cfg.extreme_q, hot_thr=hot_thr, out_weights=list(cfg.out_weights),
                seed=cfg.seed, best_ep=best_ep, scheduler=scheduler, ema_decay=ema_decay,
                window=getattr(cfg, "_window", None),
                val=val_s, test=test_s, elapsed_min=(time.time() - t0) / 60.0)
    if topk:   # SAVE_TOPK: per-tag MEAN of the k best epochs' predictions (+ topk suite)
        def _topk_mean(dfs_k):
            pcols = [c for c in dfs_k[0].columns if c.startswith("pred_m1_")]
            out = dfs_k[0].drop(columns=["tag_dt"], errors="ignore").copy()
            for d in dfs_k[1:]:
                assert list(d["tag"]) == list(out["tag"]), "topk pred tables misaligned"
            out[pcols] = np.mean([d[pcols].to_numpy(np.float32) for d in dfs_k], axis=0)
            return out
        eps_k = sorted(topk)
        tk_val = _topk_mean([topk[e][1] for e in eps_k])
        tk_test = _topk_mean([topk[e][2] for e in eps_k])
        tk_test.to_csv(os.path.join(run_dir, "preds_test_topk.csv"), index=False)
        tk_val.to_csv(os.path.join(run_dir, "preds_val_topk.csv"), index=False)
        meta["topk_eps"] = eps_k
        meta["topk_scores"] = {int(e): float(topk[e][0]) for e in eps_k}
        meta["test_topk"] = suite(tk_test, hot_thr)
        print(f"[topk] k={len(eps_k)} eps={eps_k} -> preds_{{test,val}}_topk.csv | "
              f"TEST(topk-mean) hot_all={meta['test_topk']['mae_hot_all']:.3f} "
              f"hot_ext={meta['test_topk']['mae_hot_ext']:.3f}")
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=2)
    if diag_rows:
        import pandas as pd
        pd.concat(diag_rows, ignore_index=True).to_csv(
            os.path.join(run_dir, "diag_epoch_preds.csv.gz"), index=False, compression="gzip")
        print(f"[diag] per-epoch val+test predictions -> {run_dir}/diag_epoch_preds.csv.gz")
    print(f"[DONE] {run_id} | TEST hot_all={test_s['mae_hot_all']:.3f} hot_ext={test_s['mae_hot_ext']:.3f} "
          f"| val_select={best_val:.4f}@{best_ep} | {meta['elapsed_min']:.1f}m")
    return meta


# ============================================================
# COLD (winter / MIN targets) — opt-in via COLD=1. Faithful to
# run_grid_cold_winter.py; the hot path above is untouched.
# ============================================================
def build_cold_data(cfg, cache=False, geo=None, tabular=False, y_override=None):
    """Loaders/ns/train_cold_sorted/cold_thr for the MIN (winter) dataset.

    Naming differs from hot: norm = norm_stats_extremes_full_MIN.npz, splits =
    split_{train,val,test}.csv (generic, NO _yNONE_v1 tag). Reuses data.py's
    FullSampleDataset (unchanged) with a manually loaded ns + split frames.
    train_cold_sorted = sorted TRAIN y[:,0] (output 0 = coldest) — NO leakage.
    cold_thr = TRAIN p10 (low tail), used for eval + selection only (not in loss)."""
    import pandas as pd
    from pipeline.data import FullSampleDataset, load_norm_stats  # reuse hot loaders (no edit)
    from torch.utils.data import DataLoader

    d = cfg.dataset_dir
    # norm: MIN file (fallback to the tag path via load_norm_stats if a MIN variant is absent)
    min_norm = os.path.join(d, "norm_stats_extremes_full_MIN.npz")
    if os.path.exists(min_norm):
        z = np.load(min_norm, allow_pickle=True)
        ns = dict(x_mean=z["x_mean"].astype(np.float32), x_std=z["x_std"].astype(np.float32),
                  y_median=z["y_median"].astype(np.float32), y_iqr=z["y_iqr"].astype(np.float32),
                  y_transform=str(z["y_transform"][0]),
                  T=int(z["T"]), C=int(z["C"]), H=int(z["H"]), W=int(z["W"]))
    else:
        ns = load_norm_stats(d, "MIN")   # hot-style tag fallback

    def _load_split(split):
        # generic split_<split>.csv first; fall back to the hot-style tagged name
        p = os.path.join(d, f"split_{split}.csv")
        if not os.path.exists(p):
            p = os.path.join(d, f"split_{split}_{cfg.tag_norm}.csv")
        df = pd.read_csv(p)
        df["tag"] = pd.to_datetime(df["tag"])
        df = df[df["tag"].dt.day.isin(set(cfg.keep_dom))].copy()
        return df.sort_values("tag").reset_index(drop=True)

    dfs = {s: _load_split(s) for s in ["train", "val", "test"]}
    if y_override is not None:   # TARGETS override: drop uncovered tags + y stats from override
        keys = {str(t)[:10] for t in y_override}
        n0 = {s: len(dfs[s]) for s in dfs}
        dfs = {s: dfs[s][dfs[s]["tag"].dt.strftime("%Y-%m-%d").isin(keys)].reset_index(drop=True)
               for s in dfs}
        drops = {s: n0[s] - len(dfs[s]) for s in dfs if n0[s] != len(dfs[s])}
        if drops:
            print(f"[cold data] y_override: dropped tags without override targets {drops}")
        ov = {str(t)[:10]: v for t, v in y_override.items()}
        Ytr = np.stack([ov[t] for t in dfs["train"]["tag"].dt.strftime("%Y-%m-%d")]).astype(np.float32)
        ns = dict(ns)
        ns["y_median"] = np.median(Ytr, 0).astype(np.float32)
        ns["y_iqr"] = (np.quantile(Ytr, 0.75, 0) - np.quantile(Ytr, 0.25, 0)).astype(np.float32)
        y_override = ov
    ds = {s: FullSampleDataset(dfs[s], d, ns, cfg.keep_dom, history_window=None,
                               y_override=y_override, cache=cache, geo=geo, tabular=tabular, lapse=None)
          for s in dfs}

    # TRAIN cold distribution (output 0 only) — ECDF weights + p10 threshold, no leakage
    if y_override is not None:
        train_cold_raw = np.array([y_override[t][0] for t in
                                   dfs["train"]["tag"].dt.strftime("%Y-%m-%d")], dtype=np.float32)
    else:
        train_cold_raw = np.array([float(np.load(p, allow_pickle=True)["y"].astype(np.float32)[0])
                                   for p in ds["train"].df["path"].tolist()], dtype=np.float32)
    train_cold_sorted = np.sort(train_cold_raw)
    cold_thr = float(np.quantile(train_cold_raw, cfg.extreme_q))   # p10 (low tail)

    nw = 0 if cache else cfg.num_workers
    def dl(split, shuffle):
        return DataLoader(ds[split], batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
    loaders = {"train": dl("train", True), "val": dl("val", False), "test": dl("test", False)}
    print(f"[cold data] train/val/test={len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])} "
          f"| cold_thr(p{int(cfg.extreme_q*100)})={cold_thr:.3f} (lower=colder) "
          f"| T={ns['T']} C={ns['C']} {ns['H']}x{ns['W']}")
    return loaders, ns, train_cold_sorted, cold_thr


def run_one_config_cold(cfg, loaders, ns, train_cold_sorted, cold_thr, k=2.0,
                        pretrained_path=None, geo_chans=0, warmup=0, deterministic=False,
                        tabular=False, verbose=True, save_topk=0, diag_test=False):
    """Cold (MIN) training — 1:1 with run_grid_cold_winter.py's run_one_config:
    PercentileWeighted loss, geometric out_weights, cosine LR + EMA-of-val-score
    smoothing, p10 extreme, cold selection metrics. Same model + reused knobs
    (deterministic/warmup/pretrained/geo/backbone_pool/name/spatial_stage/...)."""
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    in_chans = ns["C"] + int(geo_chans); T = ns["T"]

    if tabular:
        model = HeadOnlyModel(F=ns["C"], T=T, head_type=cfg.head_type, d_model=cfg.d_model,
                              nhead=cfg.nhead, layers=cfg.layers, dropout=cfg.dropout,
                              out_dim=cfg.out_dim, head_layers=cfg.head_layers).to(DEVICE)
    else:
        model = ConvNeXtTiny_WithHead(
            in_chans=in_chans, T=T, head_type=cfg.head_type, backbone_chunk=cfg.backbone_chunk,
            pretrained=cfg.pretrained, d_model=cfg.d_model, nhead=cfg.nhead, layers=cfg.layers,
            dropout=cfg.dropout, use_out_affine=cfg.use_out_affine, out_dim=cfg.out_dim,
            backbone_pool=cfg.backbone_pool, backbone_name=cfg.backbone_name, head_layers=cfg.head_layers,
            spatial_stage=cfg.spatial_stage,
        ).to(DEVICE)
    if pretrained_path:
        ck = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
        miss, unexp = model.backbone.m.load_state_dict(ck["enc_state_dict"], strict=False)
        print(f"[in-domain pretrain] loaded {pretrained_path} into backbone (missing={len(miss)} unexpected={len(unexp)})")

    base = PercentileWeightedMAE_Min(
        y_median=ns["y_median"], y_iqr=ns["y_iqr"], y_transform=ns["y_transform"],
        out_weights=cfg.out_weights, train_cold_sorted=train_cold_sorted,
        k=float(k), alpha=cfg.under_alpha_ext).to(DEVICE)
    loss_fn = (PercentileWeightedHinge_Min(base, beta_hot=cfg.beta_hot).to(DEVICE)
               if cfg.use_extra_hinge else base)
    if os.environ.get("PINBALL"):     # opt-in tail-quantile term on the anchor (coldest) output
        from pipeline.loss import PinballAnchor
        loss_fn = PinballAnchor(loss_fn, tau=float(os.environ["PINBALL"]),
                                weight=float(os.environ.get("PINBALL_W", "1.0"))).to(DEVICE)

    def _is_no_decay(name, p):
        return cfg.no_wd_norm and (p.ndim <= 1 or name.endswith(".bias"))
    groups = {"bb_d": [], "bb_n": [], "hd_d": [], "hd_n": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        side = "bb" if name.startswith("backbone.") else "hd"
        groups[f"{side}_{'n' if _is_no_decay(name, p) else 'd'}"].append(p)
    opt = torch.optim.AdamW(
        [{"params": groups["bb_d"], "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
         {"params": groups["bb_n"], "lr": cfg.lr_backbone, "weight_decay": 0.0},
         {"params": groups["hd_d"], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
         {"params": groups["hd_n"], "lr": cfg.lr_head, "weight_decay": 0.0}])
    # cold recipe: cosine LR + EMA-of-val-score smoothing are ALWAYS on (faithful to source)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_val, best_ep, best_state, bad = float("inf"), -1, None, 0
    ema_score = None
    topk = {}    # SAVE_TOPK: {ep: (score, df_val, df_test)} for the current k best epochs
    diag_rows = []
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for X, y_norm, y_raw, tags in loaders["train"]:
            X = X.to(DEVICE, non_blocking=True).float()
            y_raw = y_raw.to(DEVICE, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                loss = loss_fn(model(X).float(), y_raw.float())
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
        sch.step()

        df_val = collect_pred_table(model, loaders["val"], ns, DEVICE, cfg.use_amp)
        _sel = getattr(cfg, "select", "")
        raw_score = (val_select_score_component(df_val, component=_sel, cold_thr=cold_thr, low=True)
                     if _sel in ("all", "ext") else val_select_score_cold(df_val, cold_thr))
        ema_score = (COLD_EMA_ALPHA_SCORE * raw_score + (1.0 - COLD_EMA_ALPHA_SCORE) * ema_score
                     if ema_score is not None else raw_score)
        df_te_ep = None
        if diag_test:   # val<->test correlation study: store EVERY epoch's predictions
            df_te_ep = collect_pred_table(model, loaders["test"], ns, DEVICE, cfg.use_amp)
            for split, dfx in (("val", df_val), ("test", df_te_ep)):
                d = dfx.drop(columns=["tag_dt"], errors="ignore").copy()
                d.insert(0, "ep", ep); d.insert(1, "split", split)
                diag_rows.append(d)
        if save_topk and ep >= warmup:   # SAVE_TOPK: stash preds when this epoch enters the
            worst = max(topk, key=lambda e: topk[e][0]) if topk else None   # current k best
            if len(topk) < int(save_topk) or ema_score < topk[worst][0]:
                df_te_k = (df_te_ep if df_te_ep is not None else
                           collect_pred_table(model, loaders["test"], ns, DEVICE, cfg.use_amp))
                topk[ep] = (float(ema_score), df_val, df_te_k)
                if len(topk) > int(save_topk):
                    del topk[max(topk, key=lambda e: topk[e][0])]           # evict the worst
        if ep < warmup:
            pass
        elif ema_score < best_val - 1e-6 or best_state is None:
            best_val, best_ep = ema_score, ep
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and (ep == 1 or ep % 5 == 0 or bad == 0 or bad >= cfg.patience):
            m = compute_cold_metrics(df_val, cold_thr)
            print(f"[COLD|{cfg.head_type}|k{k:g}a{cfg.under_alpha_ext:g}b{cfg.beta_hot:g}|s{cfg.seed}] ep{ep:03d} "
                  f"mae_all={m['mae_cold_all']:.3f} mae_ext={m['mae_cold_ext']:.3f} "
                  f"tooWarm={m['too_warm_rate_ext']:.3f} n_ext={m['n_ext']} | "
                  f"sel_ema={ema_score:.4f} best={best_val:.4f}@{best_ep} bad={bad}/{cfg.patience}")
        if bad >= cfg.patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    df_val = collect_pred_table(model, loaders["val"], ns, DEVICE, cfg.use_amp)
    df_test = collect_pred_table(model, loaders["test"], ns, DEVICE, cfg.use_amp)
    val_s, test_s = suite_cold(df_val, cold_thr), suite_cold(df_test, cold_thr)

    os.makedirs(cfg.ckpt_root, exist_ok=True)
    run_id = f"{cfg.run_tag}__COLD__{cfg.head_type}__k{k:g}__a{cfg.under_alpha_ext:g}__b{cfg.beta_hot:g}__s{cfg.seed}"
    run_dir = os.path.join(cfg.ckpt_root, run_id); os.makedirs(run_dir, exist_ok=True)
    if os.environ.get("SAVE_CKPT", "0") == "1" and best_state is not None:   # persist weights
        torch.save({"state_dict": best_state, "best_ep": best_ep, "run_id": run_id},
                   os.path.join(run_dir, "best.pt"))
    df_test.drop(columns=["tag_dt"]).to_csv(os.path.join(run_dir, "preds_test.csv"), index=False)
    df_val.drop(columns=["tag_dt"]).to_csv(os.path.join(run_dir, "preds_val.csv"), index=False)
    meta = dict(run_id=run_id, region_dir=cfg.dataset_dir, mode="COLD", head_type=cfg.head_type,
                k=float(k), under_alpha_ext=cfg.under_alpha_ext, beta_hot=cfg.beta_hot, dropout=cfg.dropout,
                extreme_q=cfg.extreme_q, cold_thr=cold_thr, out_weights=list(cfg.out_weights),
                seed=cfg.seed, best_ep=best_ep, scheduler="cosine", ema_score_alpha=COLD_EMA_ALPHA_SCORE,
                selection="avg(val_mae_cold_all, val_mae_cold_ext) output 0 (EMA-smoothed)",
                val=val_s, test=test_s, elapsed_min=(time.time() - t0) / 60.0)
    if topk:   # SAVE_TOPK: per-tag MEAN of the k best epochs' predictions (+ topk suite)
        def _topk_mean(dfs_k):
            pcols = [c for c in dfs_k[0].columns if c.startswith("pred_m1_")]
            out = dfs_k[0].drop(columns=["tag_dt"], errors="ignore").copy()
            for d in dfs_k[1:]:
                assert list(d["tag"]) == list(out["tag"]), "topk pred tables misaligned"
            out[pcols] = np.mean([d[pcols].to_numpy(np.float32) for d in dfs_k], axis=0)
            return out
        eps_k = sorted(topk)
        tk_val = _topk_mean([topk[e][1] for e in eps_k])
        tk_test = _topk_mean([topk[e][2] for e in eps_k])
        tk_test.to_csv(os.path.join(run_dir, "preds_test_topk.csv"), index=False)
        tk_val.to_csv(os.path.join(run_dir, "preds_val_topk.csv"), index=False)
        meta["topk_eps"] = eps_k
        meta["topk_scores"] = {int(e): float(topk[e][0]) for e in eps_k}
        meta["test_topk"] = suite_cold(tk_test, cold_thr)
        print(f"[topk] k={len(eps_k)} eps={eps_k} -> preds_{{test,val}}_topk.csv | "
              f"TEST(topk-mean) mae_all={meta['test_topk']['mae_cold_all']:.3f} "
              f"mae_ext={meta['test_topk']['mae_cold_ext']:.3f}")
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=2)
    if diag_rows:
        import pandas as pd
        pd.concat(diag_rows, ignore_index=True).to_csv(
            os.path.join(run_dir, "diag_epoch_preds.csv.gz"), index=False, compression="gzip")
        print(f"[diag] per-epoch val+test predictions -> {run_dir}/diag_epoch_preds.csv.gz")
    print(f"[DONE] {run_id} | TEST mae_all={test_s['mae_cold_all']:.3f} mae_ext={test_s['mae_cold_ext']:.3f} "
          f"tooWarm={test_s['too_warm_rate_ext']:.3f} | val_select={best_val:.4f}@{best_ep} | {meta['elapsed_min']:.1f}m")
    return meta


def build_target_override(region, ranks):
    """{tag: raw targets} reusing series.py's window construction (frames X unchanged —
    only the prediction targets change). Each element of `ranks` is either an int r
    (1-indexed order statistic, as before) or a string 'tK' (mean of the K most
    extreme days — the top-k-mean output format)."""
    import pandas as pd, numpy as np
    sys.path.insert(0, os.path.join(_SRC, "baselines"))
    import series as S
    y = S.build_cluster_daily(region)["max_dry_temp"].dropna().sort_index()
    base = S.build_targets(y, hottest=True)
    need = max(int(str(r).lstrip("t")) for r in ranks)
    out = {}
    for _, row in base.iterrows():
        pp = pd.Timestamp(row["pred_point"])
        fut = y.reindex(pd.date_range(row["tag"], pp + pd.Timedelta(days=30), freq="D")).dropna().to_numpy(np.float32)
        s = np.sort(fut)[::-1]
        if len(s) > need - 1:
            out[row["tag"]] = np.array(
                [s[:int(str(r)[1:])].mean() if str(r).startswith("t") else s[int(r) - 1]
                 for r in ranks], np.float32)
    return out


def build_target_override_cold(region, ranks):
    """COLD (winter/MIN) counterpart of build_target_override: targets from the regional
    daily MINIMUM series, ranks counted from the COLD end ('t5' = mean of the 5 coldest
    days of the window)."""
    import pandas as pd, numpy as np
    sys.path.insert(0, os.path.join(_SRC, "baselines"))
    import series as S
    y = S.build_cluster_daily(region, cols=["min_dry_temp"])["min_dry_temp"].dropna().sort_index()
    base = S.build_targets(y, hottest=False)
    need = max(int(str(r).lstrip("t")) for r in ranks)
    out = {}
    for _, row in base.iterrows():
        pp = pd.Timestamp(row["pred_point"])
        fut = y.reindex(pd.date_range(row["tag"], pp + pd.Timedelta(days=30), freq="D")).dropna().to_numpy(np.float32)
        s = np.sort(fut)                                     # ascending: coldest first
        if len(s) > need - 1:
            out[row["tag"]] = np.array(
                [s[:int(str(r)[1:])].mean() if str(r).startswith("t") else s[int(r) - 1]
                 for r in ranks], np.float32)
    return out


# ============================================================
# TASK B — dense-series pretraining (opt-in via DENSE_PRETRAIN=<n_epochs>).
# Phase 1 trains the SAME architecture with out_dim=30 on the raw next-30-day DAILY
# cluster series (plain MAE in °C), then run_one_config(init_state=...) transfers all
# weights except the shape-mismatched final layer(s) and the normal order-stat run
# proceeds exactly as configured. Env unset -> nothing below runs.
# ============================================================
def build_dense_targets(region, cache_path=None):
    """{tag: (30,) float32 raw °C} = the daily cluster-mean series over the target window
    [tag, pred_point+30] — the SAME series/windows the order-stat targets come from
    (reuses series.py build_cluster_daily/build_targets, like build_target_override).
    Only tags whose 30 calendar days are all present are kept. Cached to `cache_path`."""
    import pandas as pd
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        out = {k: z[k].astype(np.float32) for k in z.files}
        print(f"[dense] loaded {len(out)} cached 30-day targets <- {cache_path}")
        return out
    sys.path.insert(0, os.path.join(_SRC, "baselines"))
    import series as S
    y = S.build_cluster_daily(region, cols=["max_dry_temp"])["max_dry_temp"].dropna().sort_index()
    base = S.build_targets(y, hottest=True)
    out = {}
    for _, row in base.iterrows():
        pp = pd.Timestamp(row["pred_point"])
        fut = y.reindex(pd.date_range(row["tag"], pp + pd.Timedelta(days=30), freq="D")).to_numpy(np.float32)
        if fut.shape[0] == 30 and np.isfinite(fut).all():
            out[str(row["tag"])] = fut
    if cache_path:
        np.savez_compressed(cache_path, **out)
        print(f"[dense] built {len(out)}/{len(base)} full 30-day targets -> {cache_path}")
    return out


def dense_pretrain(cfg, loaders, ns, region, n_epochs, dense_map, geo_chans=0, tabular=False):
    """TASK B phase 1. Same model family / optimizer / lr / grad-clip / AMP as the main
    phase, out_dim=30, loss = plain MAE in °C (denormalized; train-only median/IQR norm,
    transform 'none'). Samples whose tag has no full 30-day series are skipped.
    Returns the trained state_dict (transfer handled by run_one_config init_state)."""
    import pandas as pd
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    ds_tr = loaders["train"].dataset
    if hasattr(ds_tr, "df"):                    # FullSample / Synth / StationVec datasets
        tr_tags = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in ds_tr.df["tag"]]
    else:                                       # MmapDataset
        tr_tags = [str(t) for t in ds_tr.tags]
    have = [t for t in tr_tags if t in dense_map]
    if not have:
        raise RuntimeError("[dense] no train tags overlap the dense 30-day target map")
    Ytr = np.stack([dense_map[t] for t in have]).astype(np.float32)          # (Ntr,30) °C
    d_med = np.median(Ytr, 0).astype(np.float32)                             # train-only
    d_iqr = np.maximum(np.quantile(Ytr, 0.75, 0) - np.quantile(Ytr, 0.25, 0), 1e-6).astype(np.float32)

    in_chans = ns["C"] + int(geo_chans); T = ns["T"]
    if tabular:
        model = HeadOnlyModel(F=ns["C"], T=T, head_type=cfg.head_type, d_model=cfg.d_model,
                              nhead=cfg.nhead, layers=cfg.layers, dropout=cfg.dropout,
                              out_dim=30, head_layers=cfg.head_layers).to(DEVICE)
    else:
        model = ConvNeXtTiny_WithHead(
            in_chans=in_chans, T=T, head_type=cfg.head_type, backbone_chunk=cfg.backbone_chunk,
            pretrained=cfg.pretrained, d_model=cfg.d_model, nhead=cfg.nhead, layers=cfg.layers,
            dropout=cfg.dropout, use_out_affine=cfg.use_out_affine, out_dim=30,
            backbone_pool=cfg.backbone_pool, backbone_name=cfg.backbone_name,
            head_layers=cfg.head_layers, spatial_stage=cfg.spatial_stage,
        ).to(DEVICE)

    def _is_no_decay(name, p):
        return cfg.no_wd_norm and (p.ndim <= 1 or name.endswith(".bias"))
    groups = {"bb_d": [], "bb_n": [], "hd_d": [], "hd_n": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        side = "bb" if name.startswith("backbone.") else "hd"
        groups[f"{side}_{'n' if _is_no_decay(name, p) else 'd'}"].append(p)
    opt = torch.optim.AdamW(
        [{"params": groups["bb_d"], "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
         {"params": groups["bb_n"], "lr": cfg.lr_backbone, "weight_decay": 0.0},
         {"params": groups["hd_d"], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
         {"params": groups["hd_n"], "lr": cfg.lr_head, "weight_decay": 0.0}])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    med_t = torch.as_tensor(d_med, device=DEVICE)
    iqr_t = torch.as_tensor(d_iqr, device=DEVICE)

    print(f"[dense] phase 1 start: {len(have)}/{len(tr_tags)} train tags have a full 30-day "
          f"series | out_dim=30 | epochs={n_epochs}")
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        model.train(); tot = 0.0; nb = 0
        for X, y_norm, y_raw, tags in loaders["train"]:
            keep = [i for i, t in enumerate(tags) if t in dense_map]
            if not keep:
                continue
            X = X[keep].to(DEVICE, non_blocking=True).float()
            yd = torch.stack([torch.from_numpy(dense_map[tags[i]]) for i in keep]).to(DEVICE).float()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                pred_c = model(X).float() * iqr_t + med_t          # denorm (transform 'none')
                loss = (pred_c - yd).abs().mean()                  # plain MAE in °C
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            tot += float(loss) * len(keep); nb += len(keep)
        print(f"[dense] ep{ep:03d} MAE°C={tot/max(nb,1):.3f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"[dense] pretrained 30-day head for {n_epochs} epochs")
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ============================================================
# TASK C — multi-task cross-region training (opt-in via MULTITASK=1).
# One shared backbone + one full head per region (model.ConvNeXtTiny_MultiHead).
# X comes from the PRIMARY region's pipeline (plain frames, or PAPERW synth mode);
# frames are identical across regions — only y differs. Tags = per-split intersection
# of the three regions' splits. Env unset -> nothing below runs.
# ============================================================
MT_REGIONS = ["Center", "Negev", "Northwest"]


class _MultiYDataset(torch.utils.data.Dataset):
    """Wraps the primary-region dataset (X pipeline untouched) and swaps in the stacked
    per-region raw targets: returns (X, y_norm (R,4), y_raw (R,4), tag)."""
    def __init__(self, base_ds, ymap, y_med, y_iqr, y_tfms):
        self.base, self.ymap = base_ds, ymap                    # ymap {tag: (R,4) raw °C}
        self.y_med, self.y_iqr, self.y_tfms = y_med, y_iqr, y_tfms   # (R,4),(R,4),[str]*R

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        X, _, _, tag = self.base[idx]
        yr = self.ymap[str(tag)].astype(np.float32)             # (R,4)
        yn = (yr - self.y_med) / np.maximum(self.y_iqr, 1e-6)
        for ri, tfm in enumerate(self.y_tfms):
            if str(tfm) == "asinh":
                yn[ri] = np.arcsinh(yn[ri])
        return X, torch.from_numpy(yn.astype(np.float32)), torch.from_numpy(yr), tag


@torch.no_grad()
def collect_pred_table_multi(model, loader, ns_y, regions, device, use_amp=True):
    """collect_pred_table equivalent for the (B,R,4) multi-head model -> {region: df}."""
    import pandas as pd
    from pipeline.metrics import _names
    from pipeline.model import denorm_y_torch
    model.eval()
    rows = {r: [] for r in regions}
    cols = None
    for X, y_norm, y_raw, tags in loader:
        X = X.to(device, non_blocking=True).float()
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(X)                                      # (B,R,4)
        pred = pred.float()
        for ri, r in enumerate(regions):
            pr = denorm_y_torch(pred[:, ri],
                                torch.as_tensor(ns_y[r]["y_median"], device=device),
                                torch.as_tensor(ns_y[r]["y_iqr"], device=device),
                                ns_y[r]["y_transform"]).detach().cpu().numpy()
            tr = y_raw[:, ri].numpy()
            if cols is None:
                cols = _names(pr.shape[1])
            for i in range(pr.shape[0]):
                rows[r].append({"tag": tags[i],
                                **{f"true_m1_{c}": float(tr[i, j]) for j, c in enumerate(cols)},
                                **{f"pred_m1_{c}": float(pr[i, j]) for j, c in enumerate(cols)}})
    out = {}
    for r in regions:
        df = pd.DataFrame(rows[r])
        df.attrs["out_cols"] = cols
        df["tag_dt"] = pd.to_datetime(df["tag"])
        out[r] = df.sort_values("tag_dt").reset_index(drop=True)
    return out


def build_multitask_data(cfg, primary, synth_weights=None, cache=False,
                         history_window=None, limit_per_split=None):
    """Loaders + per-region norm/threshold for TASK C.
    Plain mode: X = FullSampleDataset on the PRIMARY region's frame dataset; y (R,4) read
    ONCE per tag from each region's sample npz. Synth mode (synth_weights set): X =
    SynthFrameDataset on the primary stationvec dir; y from the sibling stationvec dirs.
    Tags = per-split intersection of the three regions' splits (sizes printed).
    limit_per_split (smoke knob) caps each split AFTER the intersection report."""
    import pandas as pd
    from pipeline.data import (FullSampleDataset, SynthFrameDataset, load_norm_stats,
                               load_split, _abspath, _synth_norm_stats)
    from torch.utils.data import DataLoader
    regions, splits = MT_REGIONS, ["train", "val", "test"]
    assert primary in regions, f"REGION must be one of {regions} for MULTITASK"

    if synth_weights:                                            # ---- PAPERW synth X ----
        sv_primary = cfg.dataset_dir
        parent = os.path.dirname(os.path.normpath(sv_primary))
        dirs = {r: os.path.join(parent, f"stationvec_{r}") for r in regions}
        dfs = {r: {s: pd.read_csv(os.path.join(dirs[r], f"split_{s}.csv")) for s in splits}
               for r in regions}
        tagstr = {r: {s: [str(t) for t in dfs[r][s]["tag"]] for s in splits} for r in regions}
    else:                                                        # ---- plain frames X ----
        dirs = {r: REGION_DATASET[r] for r in regions}
        dfs = {r: {s: load_split(dirs[r], s, cfg.tag_norm, cfg.keep_dom) for s in splits}
               for r in regions}
        tagstr = {r: {s: [t.strftime("%Y-%m-%d") for t in dfs[r][s]["tag"]] for s in splits}
                  for r in regions}
    pathmap = {r: {s: dict(zip(tagstr[r][s], dfs[r][s]["path"])) for s in splits} for r in regions}

    inter = {s: sorted(set(tagstr[regions[0]][s])
                       & set(tagstr[regions[1]][s]) & set(tagstr[regions[2]][s]))
             for s in splits}
    for s in splits:
        per = "/".join(str(len(tagstr[r][s])) for r in regions)
        print(f"[multitask] split={s}: per-region {per} -> intersection {len(inter[s])}")

    # y cache: read each region's y ONCE per intersection tag -> {tag: (R,4)}
    ymap, dropped = {s: {} for s in splits}, 0
    for s in splits:
        tags_s = inter[s][:limit_per_split] if limit_per_split else inter[s]
        for t in tags_s:
            ys = []
            for r in regions:
                p = (os.path.join(dirs[r], os.path.basename(pathmap[r][s][t])) if synth_weights
                     else _abspath(dirs[r], pathmap[r][s][t]))
                try:
                    ys.append(np.load(p, allow_pickle=True)["y"].astype(np.float32))
                except Exception:
                    ys = None; break
            if ys is None:
                dropped += 1
            else:
                ymap[s][t] = np.stack(ys)                        # (R,4)
    if dropped:
        print(f"[multitask] dropped {dropped} intersection tags (unreadable y)")

    # per-region y-norm stats + hot_thr from the intersection TRAIN targets
    ns_y = {}
    for r in regions:
        if synth_weights:
            z = np.load(os.path.join(dirs[r], "norm_stats.npz"), allow_pickle=True)
            ns_y[r] = dict(y_median=z["y_median"].astype(np.float32),
                           y_iqr=z["y_iqr"].astype(np.float32),
                           y_transform=str(z["y_transform"][0]) if z["y_transform"].shape
                           else str(z["y_transform"]))
        else:
            nsr = load_norm_stats(dirs[r], cfg.tag_norm)
            ns_y[r] = dict(y_median=nsr["y_median"], y_iqr=nsr["y_iqr"],
                           y_transform=nsr["y_transform"])
    hot_thr = {}
    Ytr = np.stack([ymap["train"][t] for t in sorted(ymap["train"])])        # (N,R,4)
    for ri, r in enumerate(regions):
        hot_thr[r] = float(np.quantile(Ytr[:, ri, 0], cfg.extreme_q))
    print("[multitask] hot_thr(p%d): " % int(cfg.extreme_q * 100)
          + " ".join(f"{r}={hot_thr[r]:.3f}" for r in regions))

    # primary-region X datasets restricted to the y-cached intersection tags
    y_med = np.stack([ns_y[r]["y_median"] for r in regions])
    y_iqr = np.stack([ns_y[r]["y_iqr"] for r in regions])
    y_tfms = [ns_y[r]["y_transform"] for r in regions]
    if synth_weights:
        wz = np.load(synth_weights, allow_pickle=True)
        W = wz["W"].astype(np.float32); gh, gw = int(wz["grid_h"]), int(wz["grid_w"])
        Wt = np.ascontiguousarray(W.transpose(0, 2, 1))
        zp = np.load(os.path.join(dirs[primary], "norm_stats.npz"), allow_pickle=True)
        assert [str(x) for x in wz["station_names"]] == [str(x) for x in zp["station_names"]]
        assert [str(x) for x in wz["features"]] == [str(x) for x in zp["features"]]
        x_mean, x_std, _ = _synth_norm_stats(dirs[primary], synth_weights, Wt, gh, gw,
                                             dfs[primary]["train"], primary)
        ns = {"x_mean": x_mean, "x_std": x_std, "y_median": ns_y[primary]["y_median"],
              "y_iqr": ns_y[primary]["y_iqr"], "y_transform": ns_y[primary]["y_transform"],
              "C": W.shape[0], "T": int(zp["T"]), "H": gh, "W": gw}
        if history_window:
            print("[multitask] WINDOW is ignored in synth mode (SynthFrameDataset has no window)")
        def _base(s):
            df_s = dfs[primary][s][[str(t) in ymap[s] for t in dfs[primary][s]["tag"]]].reset_index(drop=True)
            return SynthFrameDataset(df_s, dirs[primary], ns, Wt, cache=cache)
    else:
        ns = load_norm_stats(dirs[primary], cfg.tag_norm)
        def _base(s):
            df_s = dfs[primary][s][[t.strftime("%Y-%m-%d") in ymap[s]
                                    for t in dfs[primary][s]["tag"]]].reset_index(drop=True)
            return FullSampleDataset(df_s, dirs[primary], ns, cfg.keep_dom,
                                     history_window=history_window, cache=cache)
    ds = {s: _MultiYDataset(_base(s), ymap[s], y_med, y_iqr, y_tfms) for s in splits}

    def dl(split, shuf):
        return DataLoader(ds[split], batch_size=cfg.batch_size, shuffle=shuf,
                          num_workers=0, pin_memory=True)
    loaders = {"train": dl("train", True), "val": dl("val", False), "test": dl("test", False)}
    print(f"[multitask] mode={'synth' if synth_weights else 'frames'} primary={primary} | "
          f"train/val/test={len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])} | "
          f"T={ns['T']} C={ns['C']} {ns['H']}x{ns['W']}")
    return dict(regions=regions, loaders=loaders, ns=ns, ns_y=ns_y, hot_thr=hot_thr,
                inter_sizes={s: len(inter[s]) for s in splits},
                mode="synth" if synth_weights else "frames")


def run_one_config_multitask(cfg, mt, primary, scheduler=None, pretrained_path=None,
                             warmup=0, deterministic=False, verbose=True):
    """TASK C training: loss = mean of the three regions' published losses (each with its
    own hot_thr + PUBLISHED alpha/beta). Per-epoch selection = the normal val score on the
    PRIMARY head. Saves preds_{val,test}_<region>.csv for all heads, meta with all three
    suites, and the shared backbone in pretrain enc_state_dict format (backbone_enc.pt)."""
    from pipeline.model import ConvNeXtTiny_MultiHead
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    regions, ns, ns_y, hot_thr = mt["regions"], mt["ns"], mt["ns_y"], mt["hot_thr"]
    loaders = mt["loaders"]

    model = ConvNeXtTiny_MultiHead(
        in_chans=ns["C"], T=ns["T"], head_type=cfg.head_type, n_heads=len(regions),
        backbone_chunk=cfg.backbone_chunk, pretrained=cfg.pretrained, d_model=cfg.d_model,
        nhead=cfg.nhead, layers=cfg.layers, dropout=cfg.dropout,
        use_out_affine=cfg.use_out_affine, out_dim=cfg.out_dim,
        backbone_pool=cfg.backbone_pool, backbone_name=cfg.backbone_name,
        head_layers=cfg.head_layers, spatial_stage=cfg.spatial_stage).to(DEVICE)
    if pretrained_path:
        ck = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
        miss, unexp = model.backbone.m.load_state_dict(ck["enc_state_dict"], strict=False)
        print(f"[in-domain pretrain] loaded {pretrained_path} into backbone (missing={len(miss)} unexpected={len(unexp)})")

    loss_fns = []
    for r in regions:
        pub = PUBLISHED[r]
        base = WeightedMAE_ExtremeUnderOnly_Celsius(
            y_median=ns_y[r]["y_median"], y_iqr=ns_y[r]["y_iqr"],
            y_transform=ns_y[r]["y_transform"], out_weights=cfg.out_weights,
            hot_thr=hot_thr[r], under_alpha_ext=pub["alpha"]).to(DEVICE)
        loss_fns.append(ExtremeUnderHinge(base, hot_thr=hot_thr[r], beta_hot=pub["beta_hot"]).to(DEVICE)
                        if cfg.use_extra_hinge else base)
    print("[multitask] per-region loss: " + " ".join(
        f"{r}(a{PUBLISHED[r]['alpha']:g},b{PUBLISHED[r]['beta_hot']:g})" for r in regions))

    def _is_no_decay(name, p):
        return cfg.no_wd_norm and (p.ndim <= 1 or name.endswith(".bias"))
    groups = {"bb_d": [], "bb_n": [], "hd_d": [], "hd_n": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        side = "bb" if name.startswith("backbone.") else "hd"
        groups[f"{side}_{'n' if _is_no_decay(name, p) else 'd'}"].append(p)
    opt = torch.optim.AdamW(
        [{"params": groups["bb_d"], "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
         {"params": groups["bb_n"], "lr": cfg.lr_backbone, "weight_decay": 0.0},
         {"params": groups["hd_d"], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
         {"params": groups["hd_n"], "lr": cfg.lr_head, "weight_decay": 0.0}])
    sch = None
    if scheduler == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_val, best_ep, best_state, bad = float("inf"), -1, None, 0
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for X, y_norm, y_raw, tags in loaders["train"]:
            X = X.to(DEVICE, non_blocking=True).float()
            y_raw = y_raw.to(DEVICE, non_blocking=True).float()          # (B,R,4)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                pred = model(X).float()                                  # (B,R,4)
                loss = torch.stack([loss_fns[ri](pred[:, ri], y_raw[:, ri])
                                    for ri in range(len(regions))]).mean()
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
        if sch is not None:
            sch.step()

        dfs_val = collect_pred_table_multi(model, loaders["val"], ns_y, regions, DEVICE, cfg.use_amp)
        df_val = dfs_val[primary]
        val_score = (val_select_score_paper(df_val, getattr(cfg, "select_q", 0.90))
                     if getattr(cfg, "select", "avg3") == "paper"
                     else val_select_score_avg3(df_val, hot_thr[primary]))
        if ep < warmup:
            pass
        elif val_score < best_val - 1e-6 or best_state is None:
            best_val, best_ep = val_score, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and (ep == 1 or ep % 5 == 0 or bad == 0 or bad >= cfg.patience):
            print(f"[MT|{cfg.head_type}|{primary}|s{cfg.seed}] ep{ep:03d} "
                  f"val_select={val_score:.4f} best={best_val:.4f}@{best_ep} bad={bad}/{cfg.patience}")
        if bad >= cfg.patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    dfs_val = collect_pred_table_multi(model, loaders["val"], ns_y, regions, DEVICE, cfg.use_amp)
    dfs_test = collect_pred_table_multi(model, loaders["test"], ns_y, regions, DEVICE, cfg.use_amp)
    val_s = {r: suite(dfs_val[r], hot_thr[r]) for r in regions}
    test_s = {r: suite(dfs_test[r], hot_thr[r]) for r in regions}

    os.makedirs(cfg.ckpt_root, exist_ok=True)
    run_id = f"{cfg.run_tag}__MT__{cfg.head_type}__s{cfg.seed}"
    run_dir = os.path.join(cfg.ckpt_root, run_id); os.makedirs(run_dir, exist_ok=True)
    for r in regions:
        dfs_test[r].drop(columns=["tag_dt"]).to_csv(
            os.path.join(run_dir, f"preds_test_{r}.csv"), index=False)
        dfs_val[r].drop(columns=["tag_dt"]).to_csv(
            os.path.join(run_dir, f"preds_val_{r}.csv"), index=False)
    meta = dict(run_id=run_id, mode=f"MULTITASK/{mt['mode']}", regions=regions, primary=primary,
                head_type=cfg.head_type, dropout=cfg.dropout, extreme_q=cfg.extreme_q,
                hot_thr={r: float(hot_thr[r]) for r in regions},
                published_ab={r: dict(alpha=PUBLISHED[r]["alpha"], beta=PUBLISHED[r]["beta_hot"])
                              for r in regions},
                out_weights=list(cfg.out_weights), seed=cfg.seed, best_ep=best_ep,
                scheduler=scheduler, intersection_sizes=mt["inter_sizes"],
                val=val_s, test=test_s, elapsed_min=(time.time() - t0) / 60.0)
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=2)
    # shared backbone in pretrain enc_state_dict format -> per-region PRETRAINED seeding
    bb_path = os.path.join(run_dir, "backbone_enc.pt")
    torch.save({"backbone": cfg.backbone_name, "region": "MULTITASK",
                "enc_state_dict": model.backbone.m.state_dict()}, bb_path)
    print(f"[multitask] shared backbone saved (enc_state_dict format) -> {bb_path}")
    print(f"[DONE] {run_id} | " + " | ".join(
        f"{r}: hot_all={test_s[r]['mae_hot_all']:.3f} hot_ext={test_s[r]['mae_hot_ext']:.3f}"
        for r in regions) + f" | val_select({primary})={best_val:.4f}@{best_ep} "
        f"| {meta['elapsed_min']:.1f}m")
    return meta


def main():
    region = os.environ.get("REGION", "Center")
    E = os.environ
    cold = E.get("COLD", "0") == "1"        # opt-in winter/MIN path; hot unaffected when off
    cfg = Config.for_region(region)
    if cold:
        # cold recipe defaults (env overrides below still apply on top, e.g. ALPHA/BETA/K)
        cfg.dataset_dir = REGION_DATASET_MIN[region]
        cfg.out_weights = COLD_OUT_WEIGHTS          # geometric (1, .5, .25, .125)
        cfg.extreme_q = 0.10                        # LOW tail p10 (eval/selection only)
        cfg.weight_decay = 1e-3                     # cold uses wd_bb=1e-3 (hot=1e-4)
        cfg.under_alpha_ext = 2.0                   # PercentileWeightedMAE_Min alpha default
        cfg.beta_hot = 1.0                          # PercentileWeightedHinge_Min default
    if "DATASET_DIR" in E: cfg.dataset_dir = E["DATASET_DIR"]   # e.g. IDW-frame dataset
    if "HEAD" in E:  cfg.head_type = E["HEAD"]
    if "ALPHA" in E: cfg.under_alpha_ext = float(E["ALPHA"])
    if "BETA" in E:  cfg.beta_hot = float(E["BETA"])
    if "DROPOUT" in E: cfg.dropout = float(E["DROPOUT"])
    if "LR_HEAD" in E: cfg.lr_head = float(E["LR_HEAD"])
    if "LR_BB" in E: cfg.lr_backbone = float(E["LR_BB"])
    if "SEED" in E:  cfg.seed = int(E["SEED"])
    if "EXTREME_Q" in E: cfg.extreme_q = float(E["EXTREME_Q"])
    if "EPOCHS" in E: cfg.epochs = int(E["EPOCHS"])
    if "PATIENCE" in E: cfg.patience = int(E["PATIENCE"])
    if "OUT_WEIGHTS" in E: cfg.out_weights = tuple(float(x) for x in E["OUT_WEIGHTS"].split(","))
    if "POOL" in E: cfg.backbone_pool = E["POOL"]
    if "BACKBONE" in E: cfg.backbone_name = E["BACKBONE"]
    if "D_MODEL" in E: cfg.d_model = int(E["D_MODEL"])
    if "NHEAD" in E: cfg.nhead = int(E["NHEAD"])
    if "HEAD_LAYERS" in E: cfg.head_layers = int(E["HEAD_LAYERS"])
    if "NO_WD_NORM" in E: cfg.no_wd_norm = E["NO_WD_NORM"] == "1"
    if "SPATIAL_STAGE" in E: cfg.spatial_stage = int(E["SPATIAL_STAGE"])
    if "SELECT" in E: cfg.select = E["SELECT"]          # 'avg3'(default) | 'paper' (Mean-MAE @ top10%)
    if "SELECT_Q" in E: cfg.select_q = float(E["SELECT_Q"])
    cfg.ckpt_root = E.get("CKPT_ROOT", experiments_root(region))
    cfg.run_tag = E.get("RUN_TAG", "RUN")
    window = int(E["WINDOW"]) if "WINDOW" in E else None
    cfg._window = window
    scheduler = E.get("SCHEDULER") or None
    ema = float(E["EMA"]) if "EMA" in E else None

    y_override = None
    if "TARGETS" in E:
        ranks = [x if x.startswith("t") else int(x) for x in E["TARGETS"].split(",")]
        cfg.out_dim = len(ranks)
        if "OUT_WEIGHTS" in E:                             # explicit env wins (u-grid ablation)
            cfg.out_weights = tuple(float(x) for x in E["OUT_WEIGHTS"].split(","))
            assert len(cfg.out_weights) == len(ranks), "OUT_WEIGHTS length must match TARGETS"
        else:
            cfg.out_weights = tuple(1.0 for _ in ranks)    # equal weights (published convention)
        cfg._targets = ranks
        if E.get("COLD", "0") == "1":
            y_override = build_target_override_cold(region, ranks)
        else:
            y_override = build_target_override(region, ranks)
        print(f"[targets] ranks={ranks} -> out_dim={cfg.out_dim} ({len(y_override)} tags overridden)")

    pretrained_path = E.get("PRETRAINED") or None
    cache = E.get("CACHE", "1") == "1"   # RAM-cache samples by default (kills npz I/O bottleneck)
    geo = None; geo_chans = 0
    if E.get("GEO", "0") == "1":
        geo = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "geo_channels.npy")).astype(np.float32)  # (3,H,W) elev,lat,lon
        geo_chans = geo.shape[0]
        print(f"[geo] appended {geo_chans} static channels (elev,lat,lon)")
    warmup = int(E.get("WARMUP", "0"))
    deterministic = E.get("DETERMINISTIC", "0") == "1"
    tabular = E.get("TABULAR", "0") == "1"   # non-spatial control (spatial-mean series, no ConvNeXt)
    lapse = E.get("LAPSE", "0") == "1"       # A3: add lapse-rate correction to frames (real elevation)
    hybrid = E.get("HYBRID", "0") == "1"     # frames + raw station vectors (PAPERW synth mode only)
    if hybrid:
        assert E.get("PAPERW"), "HYBRID=1 requires PAPERW=<interp weights .npz> (synth mode)"
        assert not cold and E.get("MULTITASK", "0") != "1" and not E.get("DENSE_PRETRAIN"), \
            "HYBRID=1 is only supported on the plain PAPERW run_one_config path"
    d_sv = int(E.get("D_SV", "256"))         # width of the hybrid station-vector projection
    save_topk = int(E.get("SAVE_TOPK", "0") or "0")   # >0: also save mean preds of k best epochs
    if E.get("BACKBONE3D", "0") == "1":      # opt-in joint space-time 3D encoder (SpatioTemporal3D)
        assert not cold and E.get("MULTITASK", "0") != "1" and not hybrid and not tabular \
            and E.get("STATIONVEC", "0") != "1" and not E.get("DENSE_PRETRAIN") \
            and not E.get("PRETRAINED"), \
            "BACKBONE3D=1 is only supported on the plain frames / PAPERW run_one_config path"

    if not os.path.isdir(cfg.dataset_dir):
        raise FileNotFoundError(
            f"dataset dir not found: {cfg.dataset_dir}\n"
            "  -> set DATASET_DIR=<dataset dir> (e.g. the bundled data/stationvec_<Region> "
            "for STATIONVEC/PAPERW runs) or DATA_ROOT=<full release root> for the canonical "
            "frame datasets. See README.md § Data.")

    if cold:                                  # ---- COLD (winter/MIN) opt-in path ----
        k = float(E.get("K", "2.0"))          # percentile weight k (grid swept this; class default 2.0)
        def _cold_sorted_from_dir(d):
            # sorted TRAIN y[:,0] (coldest) from a stationvec_MIN-style dir — ECDF basis, no leakage.
            # With a TARGETS override, the ECDF basis comes from the override targets instead.
            import pandas as pd
            df = pd.read_csv(os.path.join(d, "split_train.csv"))
            if y_override is not None:
                ys = [float(y_override[str(t)][0]) for t in df["tag"] if str(t) in y_override]
            else:
                ys = [float(np.load(os.path.join(d, os.path.basename(p)), allow_pickle=True)["y"][0])
                      for p in df["path"]]
            return np.sort(np.array(ys, dtype=np.float32))
        if E.get("STATIONVEC", "0") == "1":   # cold FAIR TABULAR: stationvec_MIN dir (612-dim)
            from pipeline.data import build_stationvec_data
            loaders, ns, cold_thr = build_stationvec_data(cfg, cfg.dataset_dir, y_override=y_override)
            train_cold_sorted = _cold_sorted_from_dir(cfg.dataset_dir)
            tabular = True
        elif E.get("PAPERW"):                 # cold SYNTH frames: weights x stationvec_MIN vectors
            from pipeline.data import build_synth_data
            loaders, ns, cold_thr = build_synth_data(
                cfg, cfg.dataset_dir, E["PAPERW"], cache=cache,
                masks_path=E.get("PAPERW_MASKS") or None, y_override=y_override)
            train_cold_sorted = _cold_sorted_from_dir(cfg.dataset_dir)
        else:                                 # original disk-frames cold path (y_override-aware)
            loaders, ns, train_cold_sorted, cold_thr = build_cold_data(
                cfg, cache=cache, geo=geo, tabular=tabular, y_override=y_override)
        run_one_config_cold(cfg, loaders, ns, train_cold_sorted, cold_thr, k=k,
                            pretrained_path=pretrained_path, geo_chans=geo_chans,
                            warmup=warmup, deterministic=deterministic, tabular=tabular,
                            save_topk=save_topk, diag_test=E.get("DIAG_TEST", "0") == "1")
        return

    if E.get("MULTITASK", "0") == "1":       # ---- TASK C: one backbone, 3 region heads ----
        mt = build_multitask_data(cfg, primary=region, synth_weights=E.get("PAPERW") or None,
                                  cache=cache, history_window=window,
                                  limit_per_split=int(E["MT_LIMIT"]) if E.get("MT_LIMIT") else None)
        run_one_config_multitask(cfg, mt, primary=region, scheduler=scheduler,
                                 pretrained_path=pretrained_path, warmup=warmup,
                                 deterministic=deterministic)
        return

    if E.get("PAPERW"):                      # synthesized-frame interpolation ablation (A3):
        from pipeline.data import build_synth_data      # station vectors x precomputed W ->
        loaders, ns, hot_thr = build_synth_data(        # (T,9,44,137) frames on the fly,
            cfg, cfg.dataset_dir, E["PAPERW"], cache=cache,  # normal spatial model below
            masks_path=E.get("PAPERW_MASKS") or None,   # opt-in mask-aware synthesis
            hybrid=hybrid, y_override=y_override)       # opt-in frames+sv hybrid batches
    elif E.get("STATIONVEC", "0") == "1":    # fair tabular ablation: flat all-station (180,612) vectors
        from pipeline.data import build_stationvec_data
        loaders, ns, hot_thr = build_stationvec_data(cfg, cfg.dataset_dir, y_override=y_override)
        tabular = True                       # -> HeadOnlyModel (F=612), no ConvNeXt
    else:
        loaders, ns, hot_thr = build_data(cfg, history_window=window, y_override=y_override,
                                          cache=cache, geo=geo, tabular=tabular, lapse=lapse)

    init_state = None
    if E.get("DENSE_PRETRAIN"):              # ---- TASK B: dense 30-day pretraining phase ----
        os.makedirs(cfg.ckpt_root, exist_ok=True)
        dense_map = build_dense_targets(
            region, cache_path=os.path.join(cfg.ckpt_root, f"dense_targets_{region}.npz"))
        init_state = dense_pretrain(cfg, loaders, ns, region, int(E["DENSE_PRETRAIN"]),
                                    dense_map, geo_chans=geo_chans, tabular=tabular)

    run_one_config(cfg, loaders, ns, hot_thr, scheduler=scheduler, ema_decay=ema,
                   pretrained_path=pretrained_path, geo_chans=geo_chans,
                   warmup=warmup, deterministic=deterministic, tabular=tabular,
                   diag_test=E.get("DIAG_TEST", "0") == "1", init_state=init_state,
                   hybrid=hybrid, d_sv=d_sv, save_topk=save_topk)


if __name__ == "__main__":
    main()
