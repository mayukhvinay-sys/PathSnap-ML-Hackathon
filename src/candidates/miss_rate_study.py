"""Candidate miss-rate study over radius R and top-K, on train + student_test trips.

Candidates are built once with the largest R and K; smaller settings are exact subsets
(distance <= R and rank <= K). Miss rate = share of GPS points (with a reference arc) whose
reference arc is NOT in the candidate set.

Run from the repo root:  python -m src.candidates.miss_rate_study
"""
import time

import numpy as np
import pandas as pd

from ._io import REPO_ROOT, list_trips, load_config, load_trip
from .align_labels import align_points_to_route
from .candidate_gen import build_candidates

R_GRID = [30, 50, 75, 100, 150]
K_GRID = [5, 8, 12, 16]


def trip_stats(trip, cfg):
    c = build_candidates(trip, max(R_GRID), max(K_GRID), cfg["gap_seconds"])
    ref = align_points_to_route(trip.track, trip.nodes, trip.arcs, trip.route,
                                cfg["label_max_dist_m"])
    n = len(trip.track)
    # distance and rank of the reference arc among each point's candidates (NaN if absent)
    ref_of_row = ref[c["obs_idx"].values]
    hit = c[c["arc_id"].values == ref_of_row]
    ref_dist = np.full(n, np.nan)
    ref_rank = np.full(n, np.nan)
    ref_dist[hit["obs_idx"].values] = hit["dist_m"].values
    ref_rank[hit["obs_idx"].values] = hit["rank_dist"].values
    return c, ref, ref_dist, ref_rank


def main():
    cfg = load_config("candidates.yaml")
    results = []          # per (trip, R, K)
    trip_rows = []
    t_all = time.time()
    for part in ["train", "student_test"]:
        for tid in list_trips(part):
            t0 = time.time()
            trip = load_trip(tid, part, with_route=True)
            c, ref, rd, rr = trip_stats(trip, cfg)
            n = len(trip.track)
            has_ref = ref >= 0
            for R in R_GRID:
                for K in K_GRID:
                    in_set = (c["dist_m"].values <= R) & (c["rank_dist"].values <= K)
                    hit = (rd <= R) & (rr <= K)           # NaN compares False
                    results.append((part, tid, R, K, int(has_ref.sum()),
                                    int(hit[has_ref].sum()), int(in_set.sum()), n))
            trip_rows.append((part, tid, n, len(trip.nodes), len(trip.arcs),
                              has_ref.mean(), time.time() - t0))
            print(f"{part} {tid} pts={n} nodes={len(trip.nodes)} "
                  f"ref_found={has_ref.mean():.3f} time={time.time()-t0:.1f}s", flush=True)
    df = pd.DataFrame(results, columns=["partition", "trip_id", "R", "K", "n_ref",
                                        "n_hit", "n_rows", "n_pts"])
    g = df.groupby(["R", "K"]).agg(n_ref=("n_ref", "sum"), n_hit=("n_hit", "sum"),
                                   n_rows=("n_rows", "sum"), n_pts=("n_pts", "sum")).reset_index()
    g["miss_rate"] = 1 - g["n_hit"] / g["n_ref"]
    g["cands_per_point"] = g["n_rows"] / g["n_pts"]
    out = REPO_ROOT / "results"
    g.to_csv(out / "miss_rate.csv", index=False)
    df.to_csv(out / "miss_rate_per_trip.csv", index=False)
    pd.DataFrame(trip_rows, columns=["partition", "trip_id", "n_pts", "n_nodes", "n_arcs",
                                     "ref_found_frac", "seconds"]).to_csv(
        out / "alignment_summary.csv", index=False)
    print(f"\ntotal time {time.time()-t_all:.0f}s")
    print(g.pivot(index="R", columns="K", values="miss_rate").round(4))
    print(g.pivot(index="R", columns="K", values="cands_per_point").round(1))


if __name__ == "__main__":
    main()
