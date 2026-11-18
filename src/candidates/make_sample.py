"""Build the small 3-trip sample file for teammates, and print quick statistics.

Run from the repo root:  python -m src.candidates.make_sample
"""
import time

import numpy as np
import pandas as pd

from ._io import REPO_ROOT, data_root, list_trips, load_config, load_trip
from .align_labels import add_labels
from .candidate_gen import build_candidates


def pick_trips(partition="train", n=3, seed=42):
    """Pick n small, varied trips deterministically."""
    rng = np.random.default_rng(seed)
    root = data_root() / partition
    eligible = []
    for tid in list_trips(partition):
        base = root / tid / tid
        n_gps = sum(1 for _ in open(str(base) + ".track"))
        n_nodes = sum(1 for _ in open(str(base) + ".nodes"))
        if 400 <= n_gps <= 1600 and n_nodes <= 12000:
            eligible.append(tid)
    return sorted(rng.choice(eligible, size=n, replace=False).tolist())


def main():
    cfg = load_config("candidates.yaml")
    trips = pick_trips()
    print("sample trips:", trips)
    frames, refs = [], []
    for tid in trips:
        t0 = time.time()
        trip = load_trip(tid, "train", with_route=True)
        c = build_candidates(trip, cfg["radius_m"], cfg["top_k"], cfg["gap_seconds"])
        c, ref = add_labels(c, trip, cfg["label_max_dist_m"])
        dt = time.time() - t0
        n_pts = len(trip.track)
        has_ref = ref["ref_arc"] >= 0
        hit = c.groupby("obs_idx")["label"].max().reindex(range(n_pts)).fillna(0)
        miss = 1 - hit[has_ref.values].mean()
        print(f"{tid}: pts={n_pts} nodes={len(trip.nodes)} arcs={len(trip.arcs)} "
              f"rows={len(c)} cands/pt={len(c)/n_pts:.1f} pts_no_cand="
              f"{n_pts - c['obs_idx'].nunique()} ref_found={has_ref.mean():.3f} "
              f"miss_rate={miss:.3f} time={dt:.1f}s")
        frames.append(c)
        refs.append(ref)
    out = pd.concat(frames, ignore_index=True)
    out_path = REPO_ROOT / "data" / "processed" / "sample_candidates.parquet"
    out.to_parquet(out_path, index=False)
    pd.concat(refs, ignore_index=True).to_parquet(
        REPO_ROOT / "data" / "processed" / "ref_arc_sample.parquet", index=False)
    print("saved", out_path, out.shape, f"positive rate={out['label'].mean():.3f}")
    print(out.drop(columns=["trip_id"]).describe().T[["mean", "min", "max"]].round(2))


if __name__ == "__main__":
    main()
