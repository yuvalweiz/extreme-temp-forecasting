"""
Aggregate the running experiments into reviewer-ready tables once they finish:
  - A4 loss ablation: exp03_center_hot_pat30/a*_b*/  (train_grid_hot meta)
  - A2 window ablation: exp02_frames/A2_win*/         (train_from_frames meta)
  - A3 kernel: exp02_frames/Center_{EXP,IDW}_tft/
Run anytime: prints whatever is done so far.
"""
import os, glob, json
import pandas as pd

EXP = "/home/weizyuv/article /experiments"


def _load(metas, getter):
    rows = []
    for m in metas:
        try:
            d = json.load(open(m))
            rows.append(getter(d, m))
        except Exception:
            pass
    return pd.DataFrame(rows)


def a4_loss():
    metas = glob.glob(f"{EXP}/exp03_center_hot_pat30/a*_b*/*/meta.json")
    def g(d, m):
        return dict(alpha=d.get("alpha"), beta=d.get("beta_hot"), best_ep=d.get("best_ep"),
                    val_all=d.get("val", {}).get("mae_hot_all"), val_ext=d.get("val", {}).get("mae_hot_ext"),
                    test_all=d.get("test", {}).get("mae_hot_all"), test_ext=d.get("test", {}).get("mae_hot_ext"))
    df = _load(metas, g)
    if len(df):
        # honest selection: pick config by VALIDATION, then report ITS test (no leakage)
        df["val_select"] = df[["val_all", "val_ext"]].mean(axis=1)
        sel = df.sort_values("val_select").iloc[0]
        print("\n=== A4 loss ablation (TFT, k=1, pat30) — Center hot ===")
        print(df.sort_values("val_select").round(3).to_string(index=False))
        print(f"  ({len(df)}/9 done) VAL-SELECTED cfg: a={sel.alpha} b={sel.beta} "
              f"-> REPORTED test_all={sel.test_all:.3f} test_ext={sel.test_ext:.3f}")
        print(f"  [diagnostic only, NOT for selection] best-by-test={df.test_all.min():.3f}")
    else:
        print("\n[A4] no configs finished yet")
    return df


def a2_window():
    metas = glob.glob(f"{EXP}/exp02_frames/A2_win*/meta.json")
    def g(d, m):
        return dict(window=int(os.path.basename(os.path.dirname(m)).replace("A2_win", "")),
                    best_ep=d.get("best_ep"), test_all=d.get("test_mae_all"), test_ext=d.get("test_mae_ext"))
    df = _load(metas, g)
    if len(df):
        print("\n=== A2 input-window ablation (TFT, k1a2b1, pat30) — Center hot ===")
        print(df.sort_values("window").round(3).to_string(index=False))
    else:
        print("\n[A2] no windows finished yet")
    return df


def a3_kernel():
    rows = []
    for k in ["EXP", "IDW", "EXP_V2"]:
        m = f"{EXP}/exp02_frames/Center_{k}_tft/meta.json"
        if os.path.exists(m):
            d = json.load(open(m))
            rows.append(dict(kernel=k, test_all=d.get("test_mae_all"), test_ext=d.get("test_mae_ext"), best_ep=d.get("best_ep")))
    df = pd.DataFrame(rows)
    if len(df):
        print("\n=== A3 interpolation-kernel ablation (TFT, k2a2b1) — Center hot ===")
        print(df.round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    a3_kernel(); a2_window(); a4_loss()
    print("\n(re-run this after more jobs complete)")
