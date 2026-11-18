"""Per-point reference labels from the ordered route (training and development only).

The .route file is only an ordered list of arcs, so we derive a reference arc for each GPS
point by monotone alignment: points are matched to route positions that never go backwards
(dynamic programming on point-to-arc distance). If no route arc is within max_dist_m the
reference arc is -1. This is an assumption, not independently verified ground truth.
"""
import numpy as np
import pandas as pd

from .candidate_gen import point_segment_geometry


def _cost_matrix(track, nodes, arcs, route, cap):
    """Distance (metres, capped) from each GPS point to each route arc: shape (N, M)."""
    gx, gy = track["x_m"].values, track["y_m"].values
    nx_, ny_ = nodes["x_m"].values, nodes["y_m"].values
    ra = arcs[route]
    ax, ay = nx_[ra[:, 0]], ny_[ra[:, 0]]
    bx, by = nx_[ra[:, 1]], ny_[ra[:, 1]]
    n, m = len(gx), len(route)
    cost = np.empty((n, m), dtype=np.float32)
    chunk = max(1, 4_000_000 // max(m, 1))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        px = np.repeat(gx[s:e], m)
        py = np.repeat(gy[s:e], m)
        d, _ = point_segment_geometry(px, py, np.tile(ax, e - s), np.tile(ay, e - s),
                                      np.tile(bx, e - s), np.tile(by, e - s))
        cost[s:e] = d.reshape(e - s, m)
    return cost


def align_points_to_route(track, nodes, arcs, route, max_dist_m=100.0):
    """Reference arc for each GPS point (-1 if none within max_dist_m)."""
    n, m = len(track), len(route)
    cost = _cost_matrix(track, nodes, arcs, route, max_dist_m)
    raw = cost.copy()
    capped = np.minimum(cost, max_dist_m)
    D = capped  # reuse memory; D[i, j] = best total cost with point i on route position j
    for i in range(1, n):
        D[i] += np.minimum.accumulate(D[i - 1])
    j = int(np.argmin(D[n - 1]))
    pos = np.empty(n, dtype=np.int64)
    pos[n - 1] = j
    for i in range(n - 1, 0, -1):
        j = int(np.argmin(D[i - 1, : j + 1]))
        pos[i - 1] = j
    ref = route[pos].copy()
    d_sel = raw[np.arange(n), pos]
    ref[d_sel > max_dist_m] = -1
    return ref


def add_labels(cands, trip, max_dist_m=100.0):
    """Add the label column (1 if the candidate is the reference arc). Needs trip.route.

    Returns (candidates with label, DataFrame[trip_id, obs_idx, ref_arc]).
    """
    if trip.route is None:
        raise ValueError("add_labels needs the route; load the trip with with_route=True")
    ref = align_points_to_route(trip.track, trip.nodes, trip.arcs, trip.route, max_dist_m)
    ref_df = pd.DataFrame({"trip_id": trip.trip_id,
                           "obs_idx": np.arange(len(ref), dtype=np.int32),
                           "ref_arc": ref})
    out = cands.copy()
    out["label"] = (out["arc_id"].values == ref[out["obs_idx"].values]).astype(np.int8)
    return out, ref_df
