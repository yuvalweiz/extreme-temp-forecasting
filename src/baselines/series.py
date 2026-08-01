"""
Shared non-spatial baseline harness: build the per-region daily cluster series
and the month-ahead order-statistic targets, EXACTLY as the original
`Basline Methods.ipynb` (cell 1-2). Every non-spatial baseline (SARIMAX,
Prophet, Tab-LSTM, and the P2 additions TimesFM/Chronos/Moirai/PatchTST)
consumes this so comparisons are apples-to-apples with the spatial model.

Protocol (see docs/BASELINE_PROTOCOL.md):
  cluster daily value = MEAN over cluster stations of each station's daily value
    (NaN if fewer than min_stations_per_day report that day)
  target series       = cluster_daily[max_dry_temp]  (min_dry_temp for cold)
  for pred_point with tag = pred_point+1 day, day-of-month in {1,7,14,21,28}:
    future = target over [tag, pred_point+30]
    p1,p3,p7,p15 = sorted(future, desc)[ [0,2,6,14] ]   (asc for cold)
  split: chronological  test=0.20, val=0.15 of train+val
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP

STATIONS_DIR = RP.stations_daily_dir(check=False)   # existence checked at use time
DATE_COL = "Date"
MIN_YEAR_KEEP = 2005
HISTORY_DAYS = 180
FORECAST_DAYS = 30
ALLOWED_DOM = {1, 7, 14, 21, 28}
TEST_FRAC = 0.20
VAL_FRAC_OF_TRAIN = 0.15

# Region cluster station lists (deduplicated), from each region's notebook.
CLUSTERS = {
    "Center": ["ASHDOD PORT", "ASHQELON PORT", "BEIT JIMAL", "BET DAGAN", "DOROT",
               "GAT", "HAFEZ HAYYIM", "NAHSHON", "NEGBA", "NIZZAN",
               "QEVUZAT YAVNE", "TEL AVIV COAST"],
    "Negev": ["ARAD", "ASHALIM", "AVDAT", "EZUZ", "MIZPE RAMON", "NEVATIM",
              "SEDE BOQER", "SHANI", "ZOMET HANEGEV", "LAHAV", "BESOR FARM", "HAZEVA"],
    "Northwest": ["AFEQ", "EN HAHORESH", "EN HASHOFET", "EN KARMEL", "GALED",
                  "HADERA PORT", "HAIFA REFINERIES", "HAIFA TECHNION",
                  "HAIFA UNIVERSITY", "NEWE YAAR", "ROSH HANIQRA",
                  "SHAVE ZIYYON", "ZIKHRON YAAQOV"],
}


def _station_file(name):
    return os.path.join(STATIONS_DIR, name.replace(" ", "_") + ".csv")


def build_cluster_daily(region, cols=None):
    """Cluster daily table = per-column MEAN across stations (min-station gated)."""
    RP.stations_daily_dir()      # fail fast with a helpful message if the CSVs are absent
    stations = list(dict.fromkeys(CLUSTERS[region]))  # dedupe, keep order
    files = [_station_file(s) for s in stations]
    min_stations = max(3, int(0.3 * len(files)))

    per_station = {}
    numeric = set()
    for f in files:
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        if DATE_COL not in d.columns:
            continue
        d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
        d = d.dropna(subset=[DATE_COL])
        d = d[d[DATE_COL].dt.year >= MIN_YEAR_KEEP].copy()
        d[DATE_COL] = d[DATE_COL].dt.normalize()
        numc = [c for c in d.columns if c != DATE_COL
                and pd.api.types.is_numeric_dtype(pd.to_numeric(d[c], errors="coerce"))]
        for c in numc:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        g = d.groupby(DATE_COL)[numc].mean()  # collapse intra-day (already daily)
        per_station[f] = g
        numeric.update(numc)

    numeric = sorted(c for c in numeric if cols is None or c in cols)
    all_dates = pd.date_range(min(g.index.min() for g in per_station.values()),
                              max(g.index.max() for g in per_station.values()), freq="D")
    out = {}
    for col in numeric:
        mat = pd.DataFrame(index=all_dates)
        for f, g in per_station.items():
            if col in g.columns:
                mat[f] = g[col].reindex(all_dates)
        mean = mat.mean(axis=1)
        mean[mat.notna().sum(axis=1) < min_stations] = np.nan
        out[col] = mean
    return pd.DataFrame(out, index=all_dates).sort_index()


def pick_order_stats(future, hottest=True):
    s = future.dropna().sort_values(ascending=not hottest)
    if len(s) < 15:
        return None
    return float(s.iloc[0]), float(s.iloc[2]), float(s.iloc[6]), float(s.iloc[14])


def build_targets(y_series, hottest=True):
    """Return df_all with true_m1_{hot,p3,p7,p15} + a 'split' column."""
    y_series = y_series.dropna().sort_index()
    lo = y_series.index.min() + pd.Timedelta(days=HISTORY_DAYS - 1)
    hi = y_series.index.max() - pd.Timedelta(days=FORECAST_DAYS)
    rows = []
    for pp in pd.date_range(lo, hi, freq="D"):
        tag = (pp + pd.Timedelta(days=1)).normalize()
        if tag.day not in ALLOWED_DOM:
            continue
        future = y_series.reindex(pd.date_range(tag, pp + pd.Timedelta(days=FORECAST_DAYS), freq="D"))
        got = pick_order_stats(future, hottest)
        if got is None:
            continue
        rows.append(dict(tag=tag.strftime("%Y-%m-%d"), pred_point=pp,
                         true_m1_hot=got[0], true_m1_p3=got[1],
                         true_m1_p7=got[2], true_m1_p15=got[3]))
    df = pd.DataFrame(rows).sort_values("tag").reset_index(drop=True)
    N = len(df)
    n_test = max(1, min(int(np.floor(TEST_FRAC * N)), N - 2))
    n_tv = N - n_test
    n_val = max(1, min(int(np.floor(VAL_FRAC_OF_TRAIN * n_tv)), n_tv - 1))
    n_train = n_tv - n_val
    df["split"] = (["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    return df


# ---------------------------------------------------------------- validation
if __name__ == "__main__":
    PREDS = RP.published_preds("Summer")
    region = os.environ.get("REGION", "Center")
    cd = build_cluster_daily(region, cols=["max_dry_temp"])
    df = build_targets(cd["max_dry_temp"], hottest=True)
    test = df[df.split == "test"].copy()
    print(f"[{region}] targets: train/val/test = "
          f"{(df.split=='train').sum()}/{(df.split=='val').sum()}/{(df.split=='test').sum()}")

    ref = pd.read_csv(os.path.join(PREDS, region, f"preds_prophet_{region}.csv"))
    m = test.merge(ref[["tag", "true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15"]],
                   on="tag", suffixes=("_mine", "_ref"))
    print(f"[validate] matched {len(m)} test tags vs existing targets")
    maxd = 0.0
    for c in ["true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15"]:
        d = (m[f"{c}_mine"] - m[f"{c}_ref"]).abs().max()
        maxd = max(maxd, d)
        print(f"   {c}: max|Δ| = {d:.4f}")
    print(f"[RESULT] max abs target diff = {maxd:.4f} "
          f"({'PASS — harness reproduces targets' if maxd < 1e-3 else 'CHECK'})")
