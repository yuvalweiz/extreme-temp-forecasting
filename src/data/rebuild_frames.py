"""
Rebuild the study-wide interpolated frames.

Step 1 (this file, --validate): reconstruct the grid + a stored frame using the
ORIGINAL ("as-coded") weights and check it matches the saved sample X. This
proves the pipeline before we generate anything new.

Step 2 (--generate, gated): regenerate the frame stack with the paper-accurate
kernel (interpolation.weights_paper) into a NEW directory. Originals untouched.

Canonical config (recovered from the preprocess notebook):
  stations = union of the 7 clusters that have data = 68 (matches paper)
  grid     = 44 x 137 pixel centers over the station bbox, padding 0.01 deg
  elevation= SRTM .bil DEMs (rasterio)
  channels = features_order.txt (9)
  kernel   = exponential, k_exp=1, d_eff=sqrt(d_km^2 + (0.001*k_f*dh_m)^2)
"""
import os, sys, glob
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                                         # <repo>/src
import interpolation as INT
import repo_paths as RP

STATIONS_DIR = RP.stations_daily_dir(check=False)   # NOT bundled (raw IMS derivative)
META = RP.stations_meta_csv()                       # bundled: data/stations_meta_data.csv
KVAL = RP.k_values_csv()                            # bundled: data/k_values.csv
BIL_DIR = RP.dem_bil_dir(check=False)               # NOT bundled (SRTM tiles)
RES_W, RES_H = 44, 137
PAD = 0.01

CLUSTERS = {
 "C0": ["AFULA NIR HAEMEQ","AMMIAD","AVNE ETAN","AYYELET HASHARHAR","BET ZAYDA","DEIR HANNA","EDEN FARM","ESHHAR","GAMLA","KEFAR NAHUM","MAALE GILBOA","MASSADA","SEDE ELIYYAHU","TAVOR KADOORIE","TEL YOSEF","YAVNEEL","ZEMAH"],
 "C1": ["AFEQ","EN HAHORESH","EN HASHOFET","EN KARMEL","GALED","HADERA PORT","HAIFA REFINERIES","HAIFA TECHNION","HAIFA UNIVERSITY","NEWE YAAR","ROSH HANIQRA","SHAVE ZIYYON","ZIKHRON YAAQOV"],
 "C2": ["HARASHIM","MEROM GOLAN PICMAN","ZEFAT HAR KENAAN","DAFNA","KEFAR BLUM","KEFAR GILADI"],
 "C3": ["ELAT","YOTVATA","NEOT SMADAR","PARAN"],
 "C4": ["ARAD","ASHALIM","AVDAT","EZUZ","MIZPE RAMON","NEVATIM","SEDE BOQER","SHANI","ZOMET HANEGEV","LAHAV","BESOR FARM","HAZEVA"],
 "C5": ["ASHDOD PORT","ASHQELON PORT","BEIT JIMAL","BET DAGAN","DOROT","GAT","HAFEZ HAYYIM","NAHSHON","NEGBA","NIZZAN","QEVUZAT YAVNE","TEL AVIV COAST"],
 "C6": ["HAR HARASHA","JERUSALEM CENTRE","JERUSALEM GIVAT RAM","ROSH ZURIM","ZOVA"],
}
FEATURE_AGG = {"max_dry_temp": "max", "min_dry_temp": "min",
               "max_wet_temp": "max", "min_wet_temp": "min",
               "max_heat_stress": "max", "min_heat_stress": "min"}  # else mean


def station_order():
    uniq = sorted(set(s for v in CLUSTERS.values() for s in v))
    files = {s: os.path.join(STATIONS_DIR, s.replace(" ", "_") + ".csv") for s in uniq}
    return [s for s in uniq if os.path.exists(files[s])]  # drops AYYELET HASHARHAR -> 68


def load_meta(order):
    m = pd.read_csv(META)
    m["Station"] = (m["Station"].astype(str).str.upper()
                    .str.replace("STATION_", "", regex=False).str.replace("_", " ", regex=False))
    if "height" in m.columns and "Height" not in m.columns:
        m = m.rename(columns={"height": "Height"})
    m = m.drop_duplicates("Station").set_index("Station").loc[order].reset_index()
    return (m["Latitude"].to_numpy(float), m["Longitude"].to_numpy(float),
            pd.to_numeric(m["Height"], errors="coerce").to_numpy(float))


def build_pixel_matrix(st_lat, st_lon):
    import rasterio
    min_lat, max_lat = st_lat.min() - PAD, st_lat.max() + PAD
    min_lon, max_lon = st_lon.min() - PAD, st_lon.max() + PAD
    lon_e = np.linspace(min_lon, max_lon, RES_W + 1)
    lat_e = np.linspace(min_lat, max_lat, RES_H + 1)
    lon_c = (lon_e[:-1] + lon_e[1:]) / 2
    lat_c = (lat_e[:-1] + lat_e[1:]) / 2
    pm = np.full((RES_W, RES_H, 3), np.nan, np.float32)
    pm[:, :, 0] = lon_c[:, None]
    pm[:, :, 1] = lat_c[None, :]
    lon = pm[:, :, 0].ravel(); lat = pm[:, :, 1].ravel()
    elev = np.full_like(lon, np.nan)
    for bil in glob.glob(os.path.join(BIL_DIR, "*.bil")):
        d = rasterio.open(bil)
        inb = ((lon >= d.bounds.left) & (lon <= d.bounds.right) &
               (lat >= d.bounds.bottom) & (lat <= d.bounds.top) & np.isnan(elev))
        if inb.any():
            s = np.array([v[0] for v in d.sample(list(zip(lon[inb], lat[inb])))], np.float32)
            if d.nodata is not None:
                s[s == d.nodata] = np.nan
            elev[inb] = s
        d.close()
    pm[:, :, 2] = elev.reshape(RES_W, RES_H)
    return pm


def feature_value_vector(order, feature, date):
    """Daily value per station for one feature on one date (NaN if missing)."""
    how = FEATURE_AGG.get(feature, "mean")
    vals = []
    for st in order:
        f = os.path.join(STATIONS_DIR, st.replace(" ", "_") + ".csv")
        df = pd.read_csv(f, usecols=lambda c: c in ("Date", feature))
        if feature not in df.columns:
            vals.append(np.nan); continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        s = pd.to_numeric(df.set_index("Date")[feature], errors="coerce")
        s = getattr(s.groupby(level=0), how)()
        vals.append(float(s.get(pd.Timestamp(date), np.nan)))
    return np.array(vals, float)


def kf_dict():
    k = pd.read_csv(KVAL)
    return dict(zip(k["Feature"], k["k"]))


def reconstruct_frame(pm, st_lat, st_lon, st_h, vals, k_f, mode="as_coded", gamma=4.0):
    P = pm.reshape(-1, 3)
    if mode == "as_coded":
        W = INT.weights_as_coded(P, st_lat, st_lon, st_h, k_f, kernel="exponential", k_exp=1.0)
    else:
        W = INT.weights_paper(P, st_lat, st_lon, st_h, k_f, gamma=gamma)
    obs = np.isfinite(vals)
    Wm = W[:, obs]
    Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-12)
    frame = Wm @ vals[obs]
    return frame.reshape(RES_W, RES_H)


def validate():
    order = station_order()
    print(f"[stations] {len(order)} (expect 68)")
    st_lat, st_lon, st_h = load_meta(order)
    pm = build_pixel_matrix(st_lat, st_lon)
    print(f"[grid] {pm.shape} lon[{pm[:,:,0].min():.3f},{pm[:,:,0].max():.3f}] "
          f"lat[{pm[:,:,1].min():.3f},{pm[:,:,1].max():.3f}] "
          f"elev[{np.nanmin(pm[:,:,2]):.0f},{np.nanmax(pm[:,:,2]):.0f}]")

    feats = open(RP.features_order_txt()).read().split()
    kf = kf_dict()
    # validation needs one stored frame sample -> the canonical dataset (full release)
    samp = np.load(os.path.join(RP.canonical_dataset("Center"), "sample_2005-07-01.npz"),
                   allow_pickle=True)
    X = samp["X"]; in_days = pd.to_datetime(samp["input_days"])
    date = in_days[-1]  # last input day = pred_point
    print(f"[validate] date={date.date()} channels={feats}")
    for feat in ["max_dry_temp", "tmp_air_wet"]:
        ci = feats.index(feat)
        vals = feature_value_vector(order, feat, date)
        rec = reconstruct_frame(pm, st_lat, st_lon, st_h, vals, kf.get(feat, 1.0), "as_coded")
        stored = X[-1, ci]
        d = np.abs(rec - stored)
        print(f"  {feat:14s} k_f={kf.get(feat):6.2f} | stored[{stored.min():.2f},{stored.max():.2f}] "
              f"recon[{np.nanmin(rec):.2f},{np.nanmax(rec):.2f}] | mean|Δ|={np.nanmean(d):.4f} max|Δ|={np.nanmax(d):.4f}")


if __name__ == "__main__":
    validate()
