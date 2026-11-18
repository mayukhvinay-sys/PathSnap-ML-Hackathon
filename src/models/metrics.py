"""Metrics module (contract A, owner C).

All functions take the candidates table `cands` (one row per GPS point x candidate arc, with
trip_id, obs_idx, arc_id [, label, after_gap]) and the model scores `p` (same length/order).
`ref` is the optional table data/processed/ref_arc_{partition}.parquet (trip_id, obs_idx, ref_arc;
ref_arc = -1 when absent). If `ref` is None the reference arc is taken from rows with label == 1,
and points with no positive row are then excluded (their miss cannot be seen).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss as _sk_log_loss, precision_score, recall_score

from src.models.common import KEY, ROOT, THRESHOLD


# ----------------------------------------------------------------------------- helpers
def _reference(cands: pd.DataFrame, ref: pd.DataFrame | None) -> pd.DataFrame:
    if ref is not None:
        r = ref[KEY + ["ref_arc"]]
    else:
        pos = cands.loc[cands["label"] == 1, KEY + ["arc_id"]].drop_duplicates(KEY)
        r = pos.rename(columns={"arc_id": "ref_arc"})
    return r[r["ref_arc"] >= 0]


def pick_top1(cands: pd.DataFrame, p) -> pd.DataFrame:
    """Highest-scoring candidate per GPS point -> DataFrame[trip_id, obs_idx, arc_id]."""
    df = cands[KEY + ["arc_id"]].reset_index(drop=True)
    df["p"] = np.asarray(p, dtype=float)
    idx = df.groupby(KEY, sort=False)["p"].idxmax()
    return df.loc[idx, KEY + ["arc_id"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- metrics
def top1_accuracy(cands: pd.DataFrame, p, ref: pd.DataFrame | None = None) -> float:
    """Observation-level top-1: best-scoring candidate arc == reference arc."""
    pred = pick_top1(cands, p)
    r = _reference(cands, ref)
    m = pred.merge(r, on=KEY, how="inner")
    return float((m["arc_id"] == m["ref_arc"]).mean()) if len(m) else float("nan")


def macro_f1(y_true, p, threshold: float = THRESHOLD) -> float:
    """Macro-F1 over classes {0,1} of the candidate-level decision p >= threshold."""
    return float(f1_score(y_true, (np.asarray(p) >= threshold).astype(int), average="macro", zero_division=0))


def precision_recall(y_true, p, threshold: float = THRESHOLD) -> tuple[float, float]:
    yhat = (np.asarray(p) >= threshold).astype(int)
    return (float(precision_score(y_true, yhat, zero_division=0)),
            float(recall_score(y_true, yhat, zero_division=0)))


def log_loss(y_true, p) -> float:
    return float(_sk_log_loss(y_true, np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15), labels=[0, 1]))


def conf_matrix(y_true, p, threshold: float = THRESHOLD) -> np.ndarray:
    return confusion_matrix(y_true, (np.asarray(p) >= threshold).astype(int), labels=[0, 1])


def candidate_miss_rate(cands: pd.DataFrame, ref: pd.DataFrame | None = None) -> float:
    """Share of GPS points whose reference arc is NOT among the generated candidates."""
    if ref is None and "label" not in cands:
        return float("nan")
    r = _reference(cands, ref)
    if ref is None:           # labels only see hits -> miss rate unknowable
        return float("nan")
    hit = cands.merge(r, left_on=KEY + ["arc_id"], right_on=KEY + ["ref_arc"])[KEY].drop_duplicates()
    return float(1.0 - len(hit) / len(r)) if len(r) else float("nan")


# ----------------------------------------------------------------------------- continuity
_ARC_CACHE: dict = {}
_CONT_IMPL = {"name": None}


def get_trip_arcs(trip_id: str, partition: str):
    """Arcs (n x 2 int array) of a trip, via A's loader if present, else np.loadtxt on the raw files."""
    key = (trip_id, partition)
    if key in _ARC_CACHE:
        return _ARC_CACHE[key]
    part = "student_test" if partition == "student" else partition
    arcs = None
    try:
        from src.data.loader import load_trip  # Member A (H0)
        arcs = np.asarray(load_trip(trip_id, part, with_route=False).arcs)
    except Exception:
        root = Path(os.environ.get("PATHSNAP_RAW", ROOT / "Student folder" / "Student folder"))
        f = root / part / trip_id / f"{trip_id}.arcs"
        if f.exists():
            arcs = np.loadtxt(f, dtype=int, ndmin=2)
    _ARC_CACHE[key] = arcs
    return arcs


def _continuity_fallback(pred: pd.DataFrame, after_gap: np.ndarray, arcs: np.ndarray) -> float:
    """Share of consecutive point pairs (not across a gap) whose arcs are identical or connected
    head-to-tail in the network (to_node of previous arc == from_node of next arc)."""
    a = pred["arc_id"].to_numpy()
    if len(a) < 2:
        return float("nan")
    ok = tot = 0
    n_arcs = len(arcs)
    for i in range(1, len(a)):
        if after_gap[i]:
            continue
        tot += 1
        u, v = int(a[i - 1]), int(a[i])
        if u == v or (0 <= u < n_arcs and 0 <= v < n_arcs and arcs[u, 1] == arcs[v, 0]):
            ok += 1
    return ok / tot if tot else float("nan")


def classifier_route_continuity(cands: pd.DataFrame, p, partition: str) -> float:
    """Mean over trips of route continuity of the classifier-only (argmax) arcs. NaN if arcs unavailable.
    Uses D's src.decoder.continuity.route_continuity when importable, otherwise the fallback above."""
    pred_all = pick_top1(cands, p)
    gap = cands.groupby(KEY, sort=False)["after_gap"].first().reset_index() if "after_gap" in cands else None
    vals = []
    for tid, pred in pred_all.groupby("trip_id", sort=False):
        arcs = get_trip_arcs(tid, partition)
        if arcs is None:
            continue
        pred = pred.sort_values("obs_idx").reset_index(drop=True)
        ag = (pred.merge(gap, on=KEY, how="left")["after_gap"].fillna(0).to_numpy()
              if gap is not None else np.zeros(len(pred)))
        v = None
        try:
            from src.decoder.continuity import route_continuity  # Member D
            v = float(route_continuity(pred[["obs_idx", "arc_id"]], arcs))
            _CONT_IMPL["name"] = "decoder.continuity"
        except Exception:
            v = _continuity_fallback(pred, ag, arcs)
            _CONT_IMPL["name"] = "fallback (metrics.py)"
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")
