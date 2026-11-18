"""Candidate generation and features.

For every GPS point we look for road arcs within a radius, keep the nearest K, and compute
features describing how well each arc explains the point. This module never reads the
route (ground truth), so it can be used on hidden trips by the API.
"""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree


def point_segment_geometry(px, py, ax, ay, bx, by):
    """Distance from points to segments A->B, and the position along the segment (0 to 1).

    All inputs are 1-D arrays of the same length (one entry per point/segment pair).
    """
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    safe = np.where(seg_len2 > 0, seg_len2, 1.0)
    t = ((px - ax) * dx + (py - ay) * dy) / safe
    t = np.where(seg_len2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx, qy = ax + t * dx, ay + t * dy
    dist = np.hypot(px - qx, py - qy)
    return dist, t


def gps_motion(track, gap_seconds):
    """Per GPS point: heading vector, speed and the after_gap flag.

    Neighbour points are only used if they are within gap_seconds in time. If neither
    neighbour is usable the heading is undefined (NaN) and speed is 0.
    """
    x, y, t = track["x_m"].values, track["y_m"].values, track["t"].values
    n = len(x)
    dt_prev = np.full(n, np.inf)
    dt_prev[1:] = np.diff(t)
    dt_next = np.full(n, np.inf)
    dt_next[:-1] = np.diff(t)
    ok_prev = (dt_prev > 0) & (dt_prev <= gap_seconds)
    ok_next = (dt_next > 0) & (dt_next <= gap_seconds)

    vx = np.zeros(n)
    vy = np.zeros(n)
    dts = np.zeros(n)
    px = np.zeros(n)
    py = np.zeros(n)
    px[1:], py[1:] = np.diff(x), np.diff(y)
    nx = np.zeros(n)
    ny = np.zeros(n)
    nx[:-1], ny[:-1] = np.diff(x), np.diff(y)
    vx += np.where(ok_prev, px, 0.0)
    vy += np.where(ok_prev, py, 0.0)
    dts += np.where(ok_prev, dt_prev, 0.0)
    vx += np.where(ok_next, nx, 0.0)
    vy += np.where(ok_next, ny, 0.0)
    dts += np.where(ok_next, dt_next, 0.0)

    have = dts > 0
    speed = np.where(have, np.hypot(vx, vy) / np.where(have, dts, 1.0), 0.0)
    moving = have & (np.hypot(vx, vy) >= 0.5)  # under 0.5 m of motion: heading unreliable
    hx = np.where(moving, vx, np.nan)
    hy = np.where(moving, vy, np.nan)
    after_gap = np.zeros(n, dtype=np.int8)
    after_gap[0] = 1
    after_gap[1:] = (dt_prev[1:] > gap_seconds).astype(np.int8)
    return hx, hy, speed, after_gap


def build_candidates(trip, radius_m=75.0, top_k=12, gap_seconds=30.0):
    """Candidate arcs and features for every GPS point of a trip (no labels).

    trip: object with .trip_id, .track (lon, lat, t, x_m, y_m), .nodes (x_m, y_m), .arcs.
    Returns a DataFrame with one row per (GPS point, candidate arc). Points with no arc
    within the radius have no rows.
    """
    track, nodes, arcs = trip.track, trip.nodes, trip.arcs
    nx_, ny_ = nodes["x_m"].values, nodes["y_m"].values
    ax, ay = nx_[arcs[:, 0]], ny_[arcs[:, 0]]
    bx, by = nx_[arcs[:, 1]], ny_[arcs[:, 1]]

    lines = shapely.linestrings(np.stack([np.column_stack([ax, ay]),
                                          np.column_stack([bx, by])], axis=1))
    tree = STRtree(lines)
    pts = shapely.points(track["x_m"].values, track["y_m"].values)
    obs_idx, arc_id = tree.query(pts, predicate="dwithin", distance=radius_m)
    if len(obs_idx) == 0:
        return pd.DataFrame()

    gx, gy = track["x_m"].values, track["y_m"].values
    dist, t_along = point_segment_geometry(gx[obs_idx], gy[obs_idx],
                                           ax[arc_id], ay[arc_id], bx[arc_id], by[arc_id])
    keep = dist <= radius_m
    obs_idx, arc_id, dist, t_along = obs_idx[keep], arc_id[keep], dist[keep], t_along[keep]

    # sort by point then distance, rank within point, keep top K
    order = np.lexsort((dist, obs_idx))
    obs_idx, arc_id, dist, t_along = obs_idx[order], arc_id[order], dist[order], t_along[order]
    first = np.r_[True, obs_idx[1:] != obs_idx[:-1]]
    start_of_group = np.maximum.accumulate(np.where(first, np.arange(len(obs_idx)), 0))
    rank = np.arange(len(obs_idx)) - start_of_group + 1
    sel = rank <= top_k
    obs_idx, arc_id, dist, t_along, rank = (obs_idx[sel], arc_id[sel], dist[sel],
                                            t_along[sel], rank[sel])

    counts = np.bincount(obs_idx, minlength=len(track))
    best = np.zeros(len(track))
    best[obs_idx[rank == 1]] = dist[rank == 1]

    hx, hy, speed, after_gap = gps_motion(track, gap_seconds)
    arc_dx, arc_dy = (bx - ax)[arc_id], (by - ay)[arc_id]
    arc_len = np.hypot(arc_dx, arc_dy)
    hxs, hys = hx[obs_idx], hy[obs_idx]
    denom = np.hypot(hxs, hys) * arc_len
    cos = (hxs * arc_dx + hys * arc_dy) / np.where(denom > 0, denom, 1.0)
    heading_diff = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    heading_diff = np.where(np.isnan(hxs) | (arc_len == 0), 90.0, heading_diff)  # neutral

    n_nodes = len(nodes)
    arc_keys = arcs[:, 0].astype(np.int64) * n_nodes + arcs[:, 1]
    twin_keys = arcs[:, 1].astype(np.int64) * n_nodes + arcs[:, 0]
    has_twin_all = np.isin(twin_keys, arc_keys).astype(np.int8)

    out = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": obs_idx.astype(np.int32),
        "t": track["t"].values[obs_idx],
        "x_m": gx[obs_idx], "y_m": gy[obs_idx],
        "lon": track["lon"].values[obs_idx], "lat": track["lat"].values[obs_idx],
        "arc_id": arc_id.astype(np.int64),
        "after_gap": after_gap[obs_idx],
        "dist_m": dist.astype(np.float32),
        "offset_frac": t_along.astype(np.float32),
        "arc_len_m": arc_len[:].astype(np.float32),
        "heading_diff_deg": heading_diff.astype(np.float32),
        "speed_mps": speed[obs_idx].astype(np.float32),
        "rank_dist": rank.astype(np.int16),
        "dist_gap_best": (dist - best[obs_idx]).astype(np.float32),
        "n_cands": counts[obs_idx].astype(np.int16),
        "has_twin": has_twin_all[arc_id],
    })
    return out
