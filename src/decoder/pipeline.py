"""Connect B's candidate generator and C's classifier to the HMM decoder."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .hmm import decode_trip


# Shared feature contract from configs/features.yaml (owned by B).
FEATURE_COLS = [
    "dist_m",
    "offset_frac",
    "arc_len_m",
    "heading_diff_deg",
    "speed_mps",
    "rank_dist",
    "dist_gap_best",
    "n_cands",
    "has_twin",
]


def score_candidates(cands: pd.DataFrame, model: Any | None = None) -> np.ndarray:
    """Score B's candidate rows, using distance until C's model is available."""
    if cands.empty:
        return np.empty(0, dtype=float)
    if model is None:
        # Temporary baseline scorer; HMM normalizes these within each GPS point.
        return 1.0 / (1.0 + cands["dist_m"].to_numpy(dtype=float))
    if not hasattr(model, "predict_proba"):
        raise TypeError("model must expose predict_proba(X)")
    missing = sorted(set(FEATURE_COLS) - set(cands.columns))
    if missing:
        raise ValueError(f"candidate table is missing model features: {missing}")
    probabilities = np.asarray(model.predict_proba(cands[FEATURE_COLS]))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(cands):
        raise ValueError("model.predict_proba must return one probability row per candidate")
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError("model has no positive class 1 for candidate relevance")
    return probabilities[:, classes.index(1)]


def decode_candidates(
    cands: pd.DataFrame,
    trip: Any,
    model: Any | None = None,
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Score B's candidate DataFrame and return the HMM arc sequence.

    ``trip`` must contain ``arcs`` and ``track``. Pass ``model=None`` for the
    distance baseline; pass C's fitted sklearn-style classifier to use its
    positive-class probabilities as HMM emissions.
    """
    scores = score_candidates(cands, model)
    return decode_trip(cands, scores, trip, params)


def build_and_decode_trip(
    trip: Any,
    model: Any | None = None,
    params: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run B's candidate generator followed by the scorer and HMM decoder."""
    # Lazy import keeps the decoder usable with saved candidate tables too.
    from src.candidates.candidate_gen import build_candidates

    cands = build_candidates(trip)
    predictions = decode_candidates(cands, trip, model=model, params=params)
    return cands, predictions
