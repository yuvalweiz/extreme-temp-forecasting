"""
Significance: moving-block bootstrap (B=10000, block_len 4 all / 2 ext) with
Holm-Bonferroni correction, ConvNeXt-TFT (ref) vs each baseline.

CORRECTION vs the original notebook: the pooled "ALL" rows concatenate each
region's per-sample loss differential exactly ONCE (the notebook tripled them:
n=2148=3x716, n_ext=285=3x95). Per-region rows are unchanged and still match
the published per-region numbers.

Run:  python run_significance.py
Writes: repo/results/summer_significance.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP
import eval_lib as E

PREDS = RP.published_preds("Summer")
DATASETS = {r: RP.eval_dataset(r) for r in RP.REGIONS}
REGIONS = list(DATASETS)
REF = "temporalfusion"
BASELINES = ["prophet", "sarimax", "tab_lstm", "lstm"]
OUT = RP.results_dir()
N_BOOT, SEED = int(os.environ.get("N_BOOT", "10000")), 0


def diff_table(ref_loss, base_loss):
    """Align ref & baseline per-sample losses on tag; return (diff_all, diff_ext)."""
    m = ref_loss.merge(base_loss, on="tag", suffixes=("_ref", "_base"))
    diff = (m["abs_err_base"] - m["abs_err_ref"]).to_numpy(float)
    ext = m["is_ext_ref"].to_numpy(bool)
    return diff, diff[ext]


def main():
    thr = {r: E.train_extreme_threshold(DATASETS[r], q=E.EXTREME_Q) for r in REGIONS}
    rows = []
    pooled = {}  # (target, baseline) -> (list diff_all, list diff_ext)

    for target, loader, lossfn in [("hot", E.load_region_hot, E.hot_losses),
                                   ("soft", E.load_region_soft, E.soft_losses)]:
        for r in REGIONS:
            preds = loader(PREDS, r)
            ref_loss = lossfn(preds[REF], thr[r])
            for b in BASELINES:
                if b not in preds:
                    continue
                d_all, d_ext = diff_table(ref_loss, lossfn(preds[b], thr[r]))
                md_a, lo_a, hi_a, p_a = E.moving_block_bootstrap(d_all, 4, N_BOOT, SEED)
                md_e, lo_e, hi_e, p_e = E.moving_block_bootstrap(d_ext, 2, N_BOOT, SEED)
                rows.append(dict(target=target, scope=r, baseline=b,
                                 n=len(d_all), n_ext=len(d_ext),
                                 dMAE_all=md_a, ci_all=f"[{lo_a:.3f},{hi_a:.3f}]", p_all=p_a,
                                 dMAE_ext=md_e, ci_ext=f"[{lo_e:.3f},{hi_e:.3f}]", p_ext=p_e))
                pooled.setdefault((target, b), ([], []))
                pooled[(target, b)][0].append(d_all)
                pooled[(target, b)][1].append(d_ext)

        # corrected pooled rows: concatenate each region ONCE
        for b in BASELINES:
            if (target, b) not in pooled:
                continue
            d_all = np.concatenate(pooled[(target, b)][0])
            d_ext = np.concatenate(pooled[(target, b)][1])
            md_a, lo_a, hi_a, p_a = E.moving_block_bootstrap(d_all, 4, N_BOOT, SEED)
            md_e, lo_e, hi_e, p_e = E.moving_block_bootstrap(d_ext, 2, N_BOOT, SEED)
            rows.append(dict(target=target, scope="ALL", baseline=b,
                             n=len(d_all), n_ext=len(d_ext),
                             dMAE_all=md_a, ci_all=f"[{lo_a:.3f},{hi_a:.3f}]", p_all=p_a,
                             dMAE_ext=md_e, ci_ext=f"[{lo_e:.3f},{hi_e:.3f}]", p_ext=p_e))

    df = pd.DataFrame(rows)
    # Holm across baselines within each (target, scope)
    df["p_all_holm"] = np.nan
    df["p_ext_holm"] = np.nan
    for (t, s), g in df.groupby(["target", "scope"]):
        df.loc[g.index, "p_all_holm"] = E.holm_bonferroni(g["p_all"]).values
        df.loc[g.index, "p_ext_holm"] = E.holm_bonferroni(g["p_ext"]).values

    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "summer_significance.csv"), index=False)
    show = df[df.scope == "ALL"][["target", "baseline", "n", "n_ext",
                                  "dMAE_all", "p_all_holm", "dMAE_ext", "p_ext_holm"]]
    print("=== POOLED (corrected single-pool) ConvNeXt-TFT vs baselines ===")
    with pd.option_context("display.width", 160):
        print(show.to_string(index=False))
    print(f"\n[OK] wrote {OUT}/summer_significance.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
