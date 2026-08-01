"""
Central path resolution — the ONE place that knows where data lives.
Every script resolves its inputs/outputs through this module, so the repo can be
cloned anywhere and pointed at data with two environment variables.

Data comes in two tiers (see README.md § Data):

1. BUNDLED data under ``<repo>/data/`` (small files are in the git repo itself;
   the station-vector datasets + interpolation weights come with the data release
   and are simply unpacked into ``<repo>/data/``):

     published_predictions/Summer/      published per-model prediction CSVs
     stationvec_<Region>/               hot station-vector datasets (68x9 daily vectors)
     stationvec_MIN_<Region>/           cold (winter) station-vector datasets
     interp_weights/                    station->grid weight matrices + norm caches
     dataset_meta/<Region>/             canonical dataset split CSVs + norm stats (small)
     grid_metadata.npz                  44x137 grid + station lat/lon/elev
     k_values.csv                       per-feature k_f (elevation/distance ratio) + R^2
     stations_meta_data.csv             station metadata (name/coords/height)
     features_order.txt                 the 9 feature channels, canonical order
     raw_ims/                           (empty stub) where YOUR raw IMS export goes

   Override the bundled-data location with ``REPO_DATA=/path``.

2. FULL / AUTHOR-LAYOUT data under ``DATA_ROOT`` (large; only needed for
   full-fidelity frame training, checkpoint verification, frame regeneration and
   the raw-data baselines).  Set ``DATA_ROOT=/path/to/full_release``; it defaults
   to ``$HOME`` (the author's machine layout):

     <DATA_ROOT>/Deep Learning Models/Cluster <R>/dataset_FULL_h180_next30_DOM_1_7_14_21_28/
     <DATA_ROOT>/Deep Learning Models/Cluster <R>/Winter Models/dataset_FULL_MIN_h180_next30_DOM_1_7_14_21_28/
     <DATA_ROOT>/Deep Learning Models/Cluster <R>/grid_ckpts_FINAL*/
     <DATA_ROOT>/Deep Learning Preprocess/Stations Daily Data 03_08_2025/<STATION>.csv
     <DATA_ROOT>/Deep Learning Preprocess/height_data/bil files/      (SRTM DEM tiles)
     <DATA_ROOT>/Deep Learning Preprocess/Cluster All_ISRAEL/Daily Aggregation */
     <DATA_ROOT>/Models Evaluations/Predictions VS Actuals/<Season>/

Fine-grained overrides (all optional): ART_DLM, ART_PREP, ART_PREDS, PREDS_DIR,
STATIONS_DIR, RESULTS_DIR.
"""
import os

# --------------------------------------------------------------------------- roots
# realpath -> correct repo root even when the repo is reached through a symlink
REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
REPO_DATA = os.environ.get("REPO_DATA", os.path.join(REPO, "data"))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(REPO, "results"))

DATA_ROOT = os.environ.get("DATA_ROOT",
                           os.environ.get("ART_DATA_ROOT", os.path.expanduser("~")))
DLM = os.environ.get("ART_DLM", os.path.join(DATA_ROOT, "Deep Learning Models"))
PREP = os.environ.get("ART_PREP", os.path.join(DATA_ROOT, "Deep Learning Preprocess"))

REGIONS = ["Center", "Negev", "Northwest"]

_DATASET_STEM = "dataset_FULL_h180_next30_DOM_1_7_14_21_28"
_DATASET_STEM_MIN = "dataset_FULL_MIN_h180_next30_DOM_1_7_14_21_28"


# --------------------------------------------------------------------------- helpers
def _require(path, what, hint):
    if path and os.path.exists(path):
        return path
    where = f" at: {path}" if path else " (none of its default locations exist)"
    raise FileNotFoundError(
        f"{what} not found{where}\n  -> {hint}\n"
        f"  (paths are resolved by src/repo_paths.py; see README.md § Data)")


def _first_existing(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def results_dir():
    os.makedirs(RESULTS, exist_ok=True)
    return RESULTS


# --------------------------------------------------------------------------- datasets
def canonical_dataset(region, cold=False, check=True):
    """The full per-sample FRAME dataset (~10 GB/region; X = 180x9x44x137)."""
    if cold:
        p = os.path.join(DLM, f"Cluster {region}", "Winter Models", _DATASET_STEM_MIN)
    else:
        p = os.path.join(DLM, f"Cluster {region}", _DATASET_STEM)
    if not check:
        return p
    return _require(p, f"canonical frame dataset ({region}{', cold' if cold else ''})",
                    "set DATA_ROOT=/path/to/full_release (or DATASET_DIR=... for training); "
                    "the frame datasets are part of the full data release, not the git repo")


def stationvec_dir(region, cold=False, check=True):
    """Bundled station-vector dataset (same y/splits as the frame dataset)."""
    p = os.path.join(REPO_DATA, ("stationvec_MIN_" if cold else "stationvec_") + region)
    if not check:
        return p
    return _require(p, f"station-vector dataset ({region}{', cold' if cold else ''})",
                    "unpack the data release into <repo>/data/ (or set REPO_DATA=/path)")


def eval_dataset(region):
    """Dataset dir used by the evaluation for TRAIN-threshold / y lookups.

    Prefers the canonical frame dataset when present; then the bundled
    station-vector dataset (identical split CSVs and per-sample y — verified:
    identical train-p90 thresholds); finally the git-bundled
    data/dataset_meta/<Region>/ whose y_values_*.csv carries the same y verbatim,
    so a bare git clone can still regenerate every reported table."""
    p = _first_existing(canonical_dataset(region, check=False),
                        stationvec_dir(region, check=False),
                        os.path.join(REPO_DATA, "dataset_meta", region))
    return _require(p, f"evaluation dataset ({region})",
                    "data/dataset_meta/<Region> ships with the git repo — is data/ intact? "
                    "(alternatively set DATA_ROOT to the full release or unpack the data "
                    "release into <repo>/data/)")


def dataset_meta(region):
    """Small canonical-dataset metadata: split_*_yNONE_v1.csv, norm stats, features_order.
    Prefers the real dataset dir; falls back to the bundled copy in data/dataset_meta/."""
    p = _first_existing(canonical_dataset(region, check=False),
                        os.path.join(REPO_DATA, "dataset_meta", region))
    return _require(p, f"dataset metadata ({region})",
                    "data/dataset_meta/<Region> ships with the git repo — is data/ intact?")


def checkpoints_dir(region, check=True):
    p = os.path.join(DLM, f"Cluster {region}", "grid_ckpts_FINAL")
    if not check:
        return p
    return _require(p, f"published checkpoints ({region})",
                    "the trained checkpoints are part of the full data release; "
                    "set DATA_ROOT=/path/to/full_release")


# --------------------------------------------------------------------------- predictions
def published_preds(season="Summer", check=True):
    """Published per-model prediction CSVs (the inputs the paper tables are computed from)."""
    env = os.environ.get("PREDS_DIR") or os.environ.get("ART_PREDS")
    cands = ([os.path.join(env, season)] if env else []) + [
        os.path.join(REPO_DATA, "published_predictions", season),
        os.path.join(DATA_ROOT, "Models Evaluations", "Predictions VS Actuals", season),
    ]
    p = _first_existing(*cands)
    if p is None and not check:
        return cands[-2]
    return _require(p, f"published prediction CSVs ({season})",
                    "data/published_predictions/ ships with the git repo — is data/ intact?")


# --------------------------------------------------------------------------- raw station data
def stations_daily_dir(check=True):
    """Per-station daily CSVs derived from the raw IMS export (see README § Data).
    Not redistributed — produced by preprocessing/01+02 notebooks from your own IMS export."""
    p = os.environ.get("STATIONS_DIR", os.path.join(PREP, "Stations Daily Data 03_08_2025"))
    if not check:
        return p
    return _require(p, "per-station daily CSV directory",
                    "download the IMS observations and run preprocessing/01_Preprocessing_"
                    "Hourly_Data.ipynb, then set DATA_ROOT (or STATIONS_DIR) accordingly")


def dem_bil_dir(check=True):
    """SRTM DEM tiles (.bil/.hdr) used to sample station/pixel elevations."""
    p = os.path.join(PREP, "height_data", "bil files")
    if not check:
        return p
    return _require(p, "SRTM DEM tile directory",
                    "download SRTM 90m tiles covering Israel (e.g. srtm_43_05/06, "
                    "srtm_44_05/06) and place the .bil/.hdr files there, or set ART_PREP")


def frames_dir(kind, check=True):
    """On-disk daily frame banks generated by preprocessing/02 (EXP | IDW | EXP_V2)."""
    names = {
        "EXP": "Daily Aggregation EXP 05_01_2026_ALL_ISRAEL_HIGHRES",
        "IDW": "Daily Aggregation IDW 05_01_2026_ALL_ISRAEL_HIGHRES",
        "EXP_V2": "Daily Aggregation EXP_V2_CANONICAL_ALL_ISRAEL_HIGHRES",
    }
    p = os.path.join(PREP, "Cluster All_ISRAEL", names[kind])
    if not check:
        return p
    return _require(p, f"frame bank ({kind})",
                    "generate the frames with preprocessing/02_Frames_and_Grid_Preprocess"
                    "_with_height_V2.ipynb (or src/data/rebuild_frames.py) or set ART_PREP")


# --------------------------------------------------------------------------- small assets
def _bundled_or(prep_rel, bundled_name, what):
    p = _first_existing(os.path.join(REPO_DATA, bundled_name),
                        os.path.join(PREP, prep_rel))
    return _require(p, what, f"data/{bundled_name} ships with the git repo — is data/ intact?")


def k_values_csv():
    return _bundled_or("k_values.csv", "k_values.csv", "k_values.csv (per-feature k_f)")


def stations_meta_csv():
    return _bundled_or("stations_meta_data.csv", "stations_meta_data.csv",
                       "stations_meta_data.csv (station coords/heights)")


def grid_metadata_npz():
    p = _first_existing(
        os.path.join(REPO_DATA, "grid_metadata.npz"),
        os.path.join(DATA_ROOT, "project codes", "hybrid_frames_all_israel", "grid_metadata.npz"))
    return _require(p, "grid_metadata.npz (grid + station coords)",
                    "data/grid_metadata.npz ships with the git repo — is data/ intact?")


def features_order_txt():
    p = _first_existing(os.path.join(REPO_DATA, "features_order.txt"),
                        os.path.join(canonical_dataset("Center", check=False), "features_order.txt"))
    return _require(p, "features_order.txt (9-channel order)",
                    "data/features_order.txt ships with the git repo — is data/ intact?")


def interp_weights_dir():
    p = os.path.join(REPO_DATA, "interp_weights")
    os.makedirs(p, exist_ok=True)
    return p


def experiments_root(region=None):
    p = os.path.join(REPO, "experiments_real")
    return os.path.join(p, region) if region else p
