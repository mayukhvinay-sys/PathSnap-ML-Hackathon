"""Shared constants and data loading for Member C's modelling code.

Data mode (env var PATHSNAP_DATA):
    full   (default) data/processed/candidates_{train,student}.parquet  (B's real files, H2)
    sample data/processed/sample_candidates.parquet (B's 3-trip file, H1): 2 trips train, 1 student
    dummy  data/processed/dummy/*  (made by tests/make_dummy_candidates.py)
Other env vars: PATHSNAP_RESULTS_DIR, PATHSNAP_ARTIFACTS_DIR (redirect outputs, used by tests).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SEED = 42            # team-wide seed
THRESHOLD = 0.5      # decision threshold for precision/recall/macro-F1 (change here only)
N_SPLITS = 5         # GroupKFold folds (grouped by trip_id)

DEFAULT_FEATURE_COLS = [
    "dist_m", "offset_frac", "arc_len_m", "heading_diff_deg", "speed_mps",
    "rank_dist", "dist_gap_best", "n_cands", "has_twin",
]
KEY = ["trip_id", "obs_idx"]


def data_mode() -> str:
    return os.environ.get("PATHSNAP_DATA", "full").lower()


def proc_dir() -> Path:
    d = ROOT / "data" / "processed"
    return d / "dummy" if data_mode() == "dummy" else d


def results_dir() -> Path:
    d = Path(os.environ.get("PATHSNAP_RESULTS_DIR", ROOT / "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def artifacts_dir() -> Path:
    d = Path(os.environ.get("PATHSNAP_ARTIFACTS_DIR", ROOT / "artifacts"))
    (d / "models").mkdir(parents=True, exist_ok=True)
    return d


def get_feature_cols() -> list[str]:
    """FEATURE_COLS from configs/features.yaml (owner: B)."""
    p = ROOT / "configs" / "features.yaml"
    if p.exists():
        cfg = yaml.safe_load(p.read_text()) or {}
        for k in ("FEATURE_COLS", "feature_cols"):
            if k in cfg and cfg[k]:
                return list(cfg[k])
    return list(DEFAULT_FEATURE_COLS)


def _fix_ids(df: pd.DataFrame) -> pd.DataFrame:
    # trip_id is an 8-digit string; keep leading zeros even if a parquet stored ints
    df = df.copy()
    df["trip_id"] = df["trip_id"].astype(str).str.zfill(8)
    return df.reset_index(drop=True)


def _read_ref(partition: str) -> pd.DataFrame | None:
    names = {"student": ["student", "student_test"], "train": ["train"]}[partition]
    for n in names:
        p = proc_dir() / f"ref_arc_{n}.parquet"
        if p.exists():
            return _fix_ids(pd.read_parquet(p))
    return None


def load_partition(partition: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (candidates, ref_arc table or None) for partition 'train' or 'student'."""
    assert partition in ("train", "student")
    mode = data_mode()
    if mode == "sample":
        p = proc_dir() / "sample_candidates.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing (Member B's H1 file). Use PATHSNAP_DATA=dummy meanwhile.")
        df = _fix_ids(pd.read_parquet(p))
        trips = sorted(df.trip_id.unique())
        keep = trips[:-1] if partition == "train" else trips[-1:]
        df = df[df.trip_id.isin(keep)].reset_index(drop=True)
        ref = None
        for n in ("sample",):
            rp = proc_dir() / f"ref_arc_{n}.parquet"
            if rp.exists():
                ref = _fix_ids(pd.read_parquet(rp))
                ref = ref[ref.trip_id.isin(keep)].reset_index(drop=True)
        return df, ref
    p = proc_dir() / f"candidates_{partition}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Expected from Member B (H2). For a dry run: "
            "python tests/make_dummy_candidates.py && set PATHSNAP_DATA=dummy")
    return _fix_ids(pd.read_parquet(p)), _read_ref(partition)


def split_xy(cands: pd.DataFrame, feature_cols: list[str] | None = None):
    feature_cols = feature_cols or get_feature_cols()
    missing = [c for c in feature_cols + ["label"] + KEY + ["arc_id"] if c not in cands.columns]
    if missing:
        raise KeyError(f"candidates table is missing columns {missing} (contract A)")
    return cands[feature_cols], cands["label"].to_numpy().astype(int), cands["trip_id"].to_numpy()
