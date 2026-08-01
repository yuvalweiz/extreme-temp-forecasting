"""
stack_weights.py — MULTI-KERNEL STACKED CHANNELS (frame-representation upgrade 1).

Concatenates existing elevation-aware interpolation weight files (compute_interp_weights.py
output: W (C0, P, S) row-stochastic over stations) along the CHANNEL axis into ONE npz, so
the synthesized frames expose the SAME station data through several kernels at once
(e.g. paper_opt + paper_g12 + as_coded_synth -> 27 channels). Preserves the elevation-aware
weight matrices unchanged — only stacks them. Output keys:
  W             (C_total, P, S)  the stacked kernels (each block byte-identical to its source)
  feature_map   (C_total,) int64 each stacked W-channel's source sv-feature index (0..C0-1,
                repeated per source file) — consumed by data.py _synth_frames /
                _synth_frames_masked: X[t,c] = W[c] @ sv[t, :, feature_map[c]]
  features      (C_total,)       channel names '<base feature>@<label>'
  base_features (C0,)            the shared per-source feature list (stationvec/masks checks)
  station_names / grid_h / grid_w  copied (validated identical across sources)
  gamma, k_f    (C_total,)       concatenated per-channel kernel params
  kernel='stacked', variant=<out stem>, source_variants/source_files/source_kernels (n_src,)
All source files must share station_names, features, grid_h/grid_w and W shape.

Usage (paths resolve against --dir, default = the repo interp_weights dir):
  python stack_weights.py paper_opt.npz paper_g12.npz as_coded_synth.npz \
      --labels opt g12 ac --out stack_opt_g12_ac.npz
"""
import argparse
import os
import numpy as np

DEFAULT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", "data", "interp_weights"))


def stack_weight_files(paths, labels, out_path):
    zs = [np.load(p, allow_pickle=True) for p in paths]
    ref = zs[0]
    stations = [str(s) for s in ref["station_names"]]
    feats = [str(f) for f in ref["features"]]
    gh, gw = int(ref["grid_h"]), int(ref["grid_w"])
    for p, z in zip(paths, zs):
        assert "feature_map" not in z.files, f"{p} is already a stacked file"
        assert [str(s) for s in z["station_names"]] == stations, f"{p}: station order mismatch"
        assert [str(f) for f in z["features"]] == feats, f"{p}: feature order mismatch"
        assert (int(z["grid_h"]), int(z["grid_w"])) == (gh, gw), f"{p}: grid mismatch"
        assert z["W"].shape == ref["W"].shape, f"{p}: W shape mismatch"

    C0 = len(feats)
    W = np.concatenate([z["W"].astype(np.float32) for z in zs], axis=0)      # (n*C0, P, S)
    feature_map = np.tile(np.arange(C0, dtype=np.int64), len(zs))            # (n*C0,)
    features = np.array([f"{f}@{lab}" for lab in labels for f in feats], dtype=object)
    gamma = np.concatenate([np.asarray(z["gamma"], np.float32).reshape(-1) if "gamma" in z.files
                            else np.full(C0, np.nan, np.float32) for z in zs])
    k_f = np.concatenate([np.asarray(z["k_f"], np.float32).reshape(-1) if "k_f" in z.files
                          else np.full(C0, np.nan, np.float32) for z in zs])
    src_var = np.array([str(z["variant"]) if "variant" in z.files
                        else os.path.splitext(os.path.basename(p))[0]
                        for p, z in zip(paths, zs)], dtype=object)
    src_ker = np.array([str(z["kernel"]) if "kernel" in z.files else "?" for z in zs],
                       dtype=object)
    out_stem = os.path.splitext(os.path.basename(out_path))[0]

    # row-stochasticity sanity (per stacked channel, over the station axis)
    rs = np.abs(W.sum(-1) - 1.0).max()
    print(f"[stack] W {W.shape} | max |row-sum - 1| = {rs:.2e}")

    np.savez(out_path,
             W=W, feature_map=feature_map, features=features,
             base_features=np.array(feats, dtype=object),
             station_names=ref["station_names"], grid_h=gh, grid_w=gw,
             gamma=gamma, k_f=k_f, kernel="stacked", variant=out_stem,
             elev=np.array([bool(z["elev"]) if "elev" in z.files else True for z in zs]),
             source_variants=src_var, source_kernels=src_ker,
             source_files=np.array([os.path.basename(p) for p in paths], dtype=object))
    print(f"[stack] {len(zs)} x {C0} channels -> {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")
    for c in range(W.shape[0]):
        if c % C0 == 0:
            print(f"  ch{c:>2}..{c + C0 - 1}: {src_var[c // C0]} "
                  f"(features '{features[c]}' .. '{features[c + C0 - 1]}')")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("inputs", nargs="+", help="source weight .npz files (>=2)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="per-source channel-name suffix (default: source variant/stem)")
    ap.add_argument("--out", required=True, help="output npz name/path")
    ap.add_argument("--dir", default=DEFAULT_DIR,
                    help=f"directory for relative paths (default {DEFAULT_DIR})")
    a = ap.parse_args()
    assert len(a.inputs) >= 2, "need at least two weight files to stack"

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(a.dir, p)

    paths = [resolve(p) for p in a.inputs]
    labels = a.labels
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    assert len(labels) == len(paths), "--labels count must match inputs"
    stack_weight_files(paths, labels, resolve(a.out))


if __name__ == "__main__":
    main()
