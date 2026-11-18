"""Prediction pipeline: candidates -> classifier scores -> HMM/Viterbi decoding.

Works on a trip loaded WITHOUT its route. It never reads reference data.
"""
import json
import time

import joblib
import numpy as np

from src.candidates.candidate_gen import build_candidates
from src.decoder.hmm import decode_trip
from src.decoder.pipeline import score_candidates

from .config import CANDIDATE_CFG, HMM_PARAMS_PATH, MODEL_PATH


class Predictor:
    def __init__(self, model_path=MODEL_PATH, params_path=HMM_PARAMS_PATH):
        self.model_path = model_path
        self.model = joblib.load(model_path)          # loaded once at start-up
        with open(params_path, "r") as f:
            self.params = json.load(f)
        self.model_name = getattr(model_path, "stem", str(model_path))

    def predict(self, trip):
        """Return per-point arcs from the classifier alone and after HMM decoding."""
        t0 = time.perf_counter()
        n = len(trip.track)
        cands = build_candidates(trip, CANDIDATE_CFG["radius_m"], CANDIDATE_CFG["top_k"],
                                 CANDIDATE_CFG["gap_seconds"])
        scores = score_candidates(cands, self.model)
        clf_arcs = np.full(n, -1, dtype=np.int64)
        if len(cands):
            tmp = cands[["obs_idx", "arc_id"]].reset_index(drop=True)
            tmp["p"] = scores
            best = tmp.loc[tmp.groupby("obs_idx")["p"].idxmax()]
            clf_arcs[best["obs_idx"].values] = best["arc_id"].values
        hmm = decode_trip(cands, scores, trip, self.params)
        hmm_arcs = np.full(n, -1, dtype=np.int64)
        hmm_arcs[hmm["obs_idx"].values] = hmm["arc_id"].values
        return {"clf_arcs": clf_arcs, "hmm_arcs": hmm_arcs, "n_candidates": int(len(cands)),
                "latency_s": time.perf_counter() - t0}
