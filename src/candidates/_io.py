"""Small private loader for raw trips (used until the shared loader is available).

Raw files per trip: <id>.track (lon lat t), <id>.nodes (lon lat), <id>.arcs (from to),
<id>.route (ordered arc ids). None of them has a header.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

REPO_ROOT = Path(__file__).resolve().parents[2]
def local_transformer(lon0, lat0):
    """Metric projection centred on a trip (azimuthal equidistant, WGS84).

    Trips are spread across Europe, so one fixed UTM zone would distort distances far from
    its central meridian. A projection centred on each trip is accurate at trip scale.
    """
    crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m"
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def load_config(name):
    with open(REPO_ROOT / "configs" / name, "r") as f:
        return yaml.safe_load(f)


def data_root():
    cfg = load_config("paths.yaml")
    return (REPO_ROOT / cfg["data_root"]).resolve()


def project_xy(lon, lat, transformer):
    """Longitude/latitude in degrees -> metres using the given local transformer."""
    x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame   # lon, lat, t, x_m, y_m
    nodes: pd.DataFrame   # lon, lat, x_m, y_m (row index = node id)
    arcs: np.ndarray      # (n_arcs, 2) int64: from_node, to_node (row index = arc id)
    route: np.ndarray     # ordered arc ids, or None when with_route=False


def list_trips(partition):
    root = data_root() / partition
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_trip(trip_id, partition, with_route=True):
    """Read one trip. With with_route=False the .route file is never opened."""
    d = data_root() / partition / trip_id
    base = d / trip_id
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
        route = np.loadtxt(str(base) + ".route", dtype=np.int64, ndmin=1)
    return Trip(trip_id, track, nodes, arcs, route)
