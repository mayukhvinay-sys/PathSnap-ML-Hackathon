from __future__ import annotations

from pathlib import Path
from typing import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


TRACK_COLUMNS = ["lon", "lat", "timestamp"]
NODE_COLUMNS = ["lon", "lat"]
ARC_COLUMNS = ["from_node", "to_node"]


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: np.ndarray
    route: np.ndarray | None


def _dataset_root() -> Path:
    """Resolve the supplied dataset root, accepting its outer extracted folder."""
    here = Path(__file__).resolve().parents[2]
    candidates = (here / "Student folder", here / "Student folder" / "Student folder")
    return next((path for path in candidates if (path / "train").is_dir()), candidates[0])


def load_track(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=TRACK_COLUMNS,
        dtype={"lon": "float64", "lat": "float64", "timestamp": "float64"},
    )


def load_nodes(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=NODE_COLUMNS,
        dtype={"lon": "float64", "lat": "float64"},
    )


def load_arcs(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ARC_COLUMNS,
        dtype={"from_node": "int64", "to_node": "int64"},
    )


def load_route(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["arc_id"])
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["arc_id"],
        dtype={"arc_id": "int64"},
    )


def trip_dirs(dataset_root: Path, partitions: Iterable[str]) -> list[Path]:
    result: list[Path] = []
    for partition in partitions:
        partition_dir = dataset_root / partition
        if not partition_dir.is_dir() and (dataset_root / "Student folder" / partition).is_dir():
            partition_dir = dataset_root / "Student folder" / partition
        if partition_dir.exists():
            result.extend(sorted(p for p in partition_dir.iterdir() if p.is_dir()))
    return result


def list_trips(partition: str, dataset_root: Path | None = None) -> list[str]:
    """Return sorted 8-digit trip identifiers for a partition."""
    if partition not in {"train", "student_test"}:
        raise ValueError("partition must be 'train' or 'student_test'")
    root = (dataset_root or _dataset_root()).resolve()
    return [p.name for p in trip_dirs(root, (partition,)) if p.name.isdigit() and len(p.name) == 8]


def load_trip(
    trip_id: str,
    partition: str,
    with_route: bool = True,
    dataset_root: Path | None = None,
) -> Trip:
    """Load and validate one trip. with_route=False never accesses the route file."""
    if partition not in {"train", "student_test"}:
        raise ValueError("partition must be 'train' or 'student_test'")
    if not trip_id.isdigit() or len(trip_id) != 8:
        raise ValueError("trip_id must be an 8-digit string (preserve leading zeros)")
    root = (dataset_root or _dataset_root()).resolve()
    trip_dir = root / partition / trip_id
    if not trip_dir.is_dir() and (root / "Student folder" / partition / trip_id).is_dir():
        trip_dir = root / "Student folder" / partition / trip_id
    required = [trip_dir / f"{trip_id}.{suffix}" for suffix in ("track", "nodes", "arcs")]
    absent = [p.name for p in required if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"Trip {trip_id} is missing required files: {', '.join(absent)}")

    track_raw = load_track(required[0])
    track_raw.rename(columns={"timestamp": "t"}, inplace=True)
    track_raw["t"] = pd.to_numeric(track_raw["t"], errors="coerce")
    nodes = load_nodes(required[1])
    arcs_df = load_arcs(required[2])
    track_xy = np.isfinite(track_raw[["lon", "lat", "t"]]).all(axis=1)
    track_coords = track_raw["lon"].between(-180, 180) & track_raw["lat"].between(-90, 90)
    if not track_xy.all() or not track_coords.all():
        raise ValueError(f"Trip {trip_id} contains missing/non-finite or out-of-range GPS values")
    node_valid = np.isfinite(nodes[["lon", "lat"]]).all(axis=1) & nodes["lon"].between(-180, 180) & nodes["lat"].between(-90, 90)
    if not node_valid.all():
        raise ValueError(f"Trip {trip_id} contains invalid road-network node coordinates")
    arcs = arcs_df.to_numpy(dtype=np.int64)
    if arcs.size and ((arcs < 0).any() or (arcs >= len(nodes)).any()):
        raise ValueError(f"Trip {trip_id} contains arc endpoints outside node index range")
    try:
        from .project import project_xy
    except ImportError:  # direct script import with src on sys.path
        from data.project import project_xy
    center_lon = float(track_raw["lon"].mean())
    center_lat = float(track_raw["lat"].mean())
    track_raw["x_m"], track_raw["y_m"] = project_xy(
        track_raw["lon"], track_raw["lat"], center_lon, center_lat
    )
    nodes["x_m"], nodes["y_m"] = project_xy(
        nodes["lon"], nodes["lat"], center_lon, center_lat
    )
    # The timestamp alias keeps existing EDA scripts compatible with the agreed `t` field.
    track_raw["timestamp"] = track_raw["t"]
    route = None
    if with_route:
        route_path = trip_dir / f"{trip_id}.route"
        if route_path.is_file():
            route = load_route(route_path)["arc_id"].to_numpy(dtype=np.int64)
            if route.size and ((route < 0).any() or (route >= len(arcs)).any()):
                raise ValueError(f"Trip {trip_id} contains route arc ids outside arc index range")
    return Trip(trip_id, partition, track_raw[["lon", "lat", "t", "x_m", "y_m", "timestamp"]], nodes.assign(node_id=np.arange(len(nodes)))[["lon", "lat", "x_m", "y_m", "node_id"]], arcs, route)


def validate_trip(trip_dir: Path) -> dict[str, object]:
    trip_id = trip_dir.name
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    missing = [
        path.name
        for path in (track_path, nodes_path, arcs_path)
        if not path.exists()
    ]
    if missing:
        return {"trip": trip_id, "valid": False, "missing": ", ".join(missing)}

    track = load_track(track_path)
    nodes = load_nodes(nodes_path)
    arcs = load_arcs(arcs_path)
    route = load_route(route_path)

    invalid_gps = (
        (~np.isfinite(track[["lon", "lat", "timestamp"]])).any(axis=1)
        | track["lon"].between(-180, 180).eq(False)
        | track["lat"].between(-90, 90).eq(False)
    )
    node_invalid = (
        (~np.isfinite(nodes[["lon", "lat"]])).any(axis=1)
        | nodes["lon"].between(-180, 180).eq(False)
        | nodes["lat"].between(-90, 90).eq(False)
    )
    valid_arc_rows = arcs["from_node"].between(0, len(nodes) - 1) & arcs[
        "to_node"
    ].between(0, len(nodes) - 1)

    arc_ids = set(range(len(arcs)))
    invalid_route_ids = (
        0
        if route.empty
        else int((~route["arc_id"].isin(arc_ids)).sum())
    )

    return {
        "trip": trip_id,
        "valid": not missing
        and not invalid_gps.any()
        and not node_invalid.any()
        and bool(valid_arc_rows.all())
        and invalid_route_ids == 0,
        "gps_points": len(track),
        "nodes": len(nodes),
        "arcs": len(arcs),
        "route_present": route_path.exists(),
        "route_arcs": len(route),
        "invalid_gps": int(invalid_gps.sum()),
        "coordinate_order_check": bool(
            track["lon"].between(-180, 180).all()
            and track["lat"].between(-90, 90).all()
            and track["lon"].max() >= nodes["lon"].min() - 0.1
            and track["lon"].min() <= nodes["lon"].max() + 0.1
            and track["lat"].max() >= nodes["lat"].min() - 0.1
            and track["lat"].min() <= nodes["lat"].max() + 0.1
        ),
        "invalid_nodes": int(node_invalid.sum()),
        "invalid_arc_endpoints": int((~valid_arc_rows).sum()),
        "invalid_route_arc_ids": invalid_route_ids,
        "track_dtypes": ", ".join(f"{col}:{track[col].dtype}" for col in TRACK_COLUMNS),
        "node_dtypes": ", ".join(f"{col}:{nodes[col].dtype}" for col in NODE_COLUMNS),
        "arc_dtypes": ", ".join(f"{col}:{arcs[col].dtype}" for col in ARC_COLUMNS),
    }
