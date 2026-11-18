"""Load a trip from an arbitrary trip folder (used by the API).

with_route=False never opens the .route file, so the prediction path cannot see the
reference route.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.candidates._io import Trip, local_transformer, project_xy


def list_trip_ids(trip_dir):
    trip_dir = Path(trip_dir)
    if not trip_dir.is_dir():
        return []
    return sorted(p.name for p in trip_dir.iterdir()
                  if p.is_dir() and (p / f"{p.name}.track").exists())


def load_trip_from_dir(trip_dir, trip_id, with_route=False):
    if trip_id not in list_trip_ids(trip_dir):   # also blocks path tricks like "../x"
        raise KeyError(trip_id)
    base = Path(trip_dir) / trip_id / trip_id
    tr = np.loadtxt(str(base) + ".track", ndmin=2)
    nd = np.loadtxt(str(base) + ".nodes", ndmin=2)
    arcs = np.loadtxt(str(base) + ".arcs", dtype=np.int64, ndmin=2)
    track = pd.DataFrame({"lon": tr[:, 0], "lat": tr[:, 1], "t": tr[:, 2]})
    tf = local_transformer(float(track["lon"].mean()), float(track["lat"].mean()))
    track["x_m"], track["y_m"] = project_xy(track["lon"].values, track["lat"].values, tf)
    nodes = pd.DataFrame({"lon": nd[:, 0], "lat": nd[:, 1]})
    nodes["x_m"], nodes["y_m"] = project_xy(nodes["lon"].values, nodes["lat"].values, tf)
    route = None
    if with_route:
        route_file = Path(str(base) + ".route")
        if route_file.exists():
            route = np.loadtxt(route_file, dtype=np.int64, ndmin=1)
    return Trip(trip_id, track, nodes, arcs, route)
