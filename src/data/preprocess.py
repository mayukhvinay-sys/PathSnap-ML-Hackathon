from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
try:
    from pyproj import Transformer
except ImportError:
    Transformer = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import ARC_COLUMNS, NODE_COLUMNS, TRACK_COLUMNS, load_arcs, load_nodes, load_track, trip_dirs


class _Utm32NTransformer:
    def __init__(self) -> None:
        self.a = 6378137.0
        self.e2 = 0.006694379990141316
        self.k0 = 0.9996
        self.lambda0 = np.radians(9.0)

    def transform(self, longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon = np.radians(np.asarray(longitude, dtype=float))
        lat = np.radians(np.asarray(latitude, dtype=float))
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        tan_lat = np.tan(lat)
        ep2 = self.e2 / (1 - self.e2)
        n = self.a / np.sqrt(1 - self.e2 * sin_lat ** 2)
        t = tan_lat ** 2
        c = ep2 * cos_lat ** 2
        a = (lon - self.lambda0) * cos_lat
        m = self.a * (
            (1 - self.e2 / 4 - 3 * self.e2 ** 2 / 64 - 5 * self.e2 ** 3 / 256) * lat
            - (3 * self.e2 / 8 + 3 * self.e2 ** 2 / 32 + 45 * self.e2 ** 3 / 1024) * np.sin(2 * lat)
            + (15 * self.e2 ** 2 / 256 + 45 * self.e2 ** 3 / 1024) * np.sin(4 * lat)
            - (35 * self.e2 ** 3 / 3072) * np.sin(6 * lat)
        )
        easting = self.k0 * n * (
            a + (1 - t + c) * a ** 3 / 6
            + (5 - 18 * t + t ** 2 + 72 * c - 58 * ep2) * a ** 5 / 120
        ) + 500000.0
        northing = self.k0 * (
            m + n * tan_lat * (
                a ** 2 / 2
                + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
                + (61 - 58 * t + t ** 2 + 600 * c - 330 * ep2) * a ** 6 / 720
            )
        )
        return easting, northing


def preprocessing_summary(
    trip_dir: Path, transformer: Transformer
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    trip_id = trip_dir.name
    track = load_track(trip_dir / f"{trip_id}.track").copy()
    nodes = load_nodes(trip_dir / f"{trip_id}.nodes").copy()
    arcs = load_arcs(trip_dir / f"{trip_id}.arcs").copy()

    original = track.copy()
    original_rows = len(track)
    track = track.replace([np.inf, -np.inf], np.nan)
    missing_rows = int(track[TRACK_COLUMNS].isna().any(axis=1).sum())
    track = track.dropna(subset=TRACK_COLUMNS)
    track = track[
        track["lon"].between(-180, 180)
        & track["lat"].between(-90, 90)
    ].copy()
    track = track.sort_values("timestamp", kind="stable").reset_index(drop=True)

    duplicate_mask = track.duplicated(
        subset=["lon", "lat", "timestamp"], keep="first"
    )
    duplicate_rows = int(duplicate_mask.sum())
    track = track.loc[~duplicate_mask].reset_index(drop=True)

    easting, northing = transformer.transform(
        track["lon"].to_numpy(), track["lat"].to_numpy()
    )
    track["x_m"] = easting
    track["y_m"] = northing

    gaps = track["timestamp"].diff()
    valid_gaps = gaps[gaps > 0].dropna()

    node_easting, node_northing = transformer.transform(
        nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    )
    nodes["x_m"] = node_easting
    nodes["y_m"] = node_northing

    raw_track = load_track(trip_dir / f"{trip_id}.track")
    raw_nodes = load_nodes(trip_dir / f"{trip_id}.nodes")
    raw_arcs = load_arcs(trip_dir / f"{trip_id}.arcs")
    invalid_coords = (
        ~raw_track["lon"].between(-180, 180)
        | ~raw_track["lat"].between(-90, 90)
    )
    summary: dict[str, float | int | str] = {
        "trip": trip_id,
        "gps_rows_original": original_rows,
        "gps_rows_clean": len(track),
        "missing_rows_removed": missing_rows,
        "invalid_coordinate_rows_removed": int(invalid_coords.sum()),
        "invalid_or_missing_removed": missing_rows + int(invalid_coords.sum()),
        "duplicate_rows_removed": duplicate_rows,
        "missing_lon": int(raw_track["lon"].isna().sum()),
        "missing_lat": int(raw_track["lat"].isna().sum()),
        "missing_timestamp": int(raw_track["timestamp"].isna().sum()),
        "invalid_node_coordinates": int((
            ~raw_nodes["lon"].between(-180, 180)
            | ~raw_nodes["lat"].between(-90, 90)
            | ~np.isfinite(raw_nodes[["lon", "lat"]]).all(axis=1)
        ).sum()),
        "invalid_arc_endpoints": int((
            ~raw_arcs["from_node"].between(0, len(raw_nodes)-1)
            | ~raw_arcs["to_node"].between(0, len(raw_nodes)-1)
        ).sum()),
        "timestamp_decreasing_before_sort": int(
            (raw_track["timestamp"].diff() < 0).sum()
        ),
        "sampling_gap_count_over_2s": int((gaps > 2).sum()),
        "max_sampling_gap_s": float(valid_gaps.max()) if not valid_gaps.empty else 0.0,
        "median_sampling_gap_s": float(valid_gaps.median()) if not valid_gaps.empty else 0.0,
        "coordinate_order_check": bool(
            track["lon"].between(-180, 180).all()
            and track["lat"].between(-90, 90).all()
            and track["lon"].max() >= nodes["lon"].min() - 0.1
            and track["lon"].min() <= nodes["lon"].max() + 0.1
            and track["lat"].max() >= nodes["lat"].min() - 0.1
            and track["lat"].min() <= nodes["lat"].max() + 0.1
        ),
        "gps_lon_min": float(track["lon"].min()) if not track.empty else np.nan,
        "gps_lon_max": float(track["lon"].max()) if not track.empty else np.nan,
        "gps_lat_min": float(track["lat"].min()) if not track.empty else np.nan,
        "gps_lat_max": float(track["lat"].max()) if not track.empty else np.nan,
    }
    return track, nodes, arcs, summary


def write_quality_markdown(
    inventory: pd.DataFrame,
    summaries: pd.DataFrame,
    gaps_by_trip: pd.DataFrame,
    output_path: Path,
) -> None:
    max_gap_trips = gaps_by_trip.sort_values(
        "max_sampling_gap_s", ascending=False
    ).head(10)
    missing_or_invalid = summaries[
        (summaries["invalid_or_missing_removed"] > 0)
        | (summaries["duplicate_rows_removed"] > 0)
    ]

    inventory_lines = [
        "| partition | trip | gps_points | nodes | arcs |",
        "|---|---|---:|---:|---:|",
    ]
    for row in inventory.itertuples(index=False):
        inventory_lines.append(
            f"| {row.partition} | {row.trip} | {row.gps_points} | "
            f"{row.nodes} | {row.arcs} |"
        )

    lines = [
        "# Data Quality Record",
        "",
        "## Preprocessing Decisions",
        "",
        "- Coordinate system: WGS84 longitude/latitude (EPSG:4326) projected to ETRS89 / UTM zone 32N (EPSG:25832).",
        "- Invalid and non-finite GPS rows were removed.",
        "- GPS observations were sorted by timestamp using a stable sort.",
        "- Exact duplicate GPS rows were removed.",
        "- Sampling gaps are the positive differences in consecutive timestamps after sorting.",
        "- A gap is flagged when it exceeds 2 seconds.",
        "- Coordinate order is longitude, latitude, matching the supplied file schema. Each trip's GPS extent is checked for overlap with its road-network extent (with a 0.1° tolerance).",
        f"- GPS/network extent overlap checks passed: {int(summaries['coordinate_order_check'].sum())} of {len(summaries)} complete trips.",
        f"- GPS longitude range: {summaries['gps_lon_min'].min():.6f} to {summaries['gps_lon_max'].max():.6f}; latitude range: {summaries['gps_lat_min'].min():.6f} to {summaries['gps_lat_max'].max():.6f}.",
        "- Projection limitation: the handoff contract requires EPSG:25832, but these trips span multiple continents. EPSG:25832 coordinates are distorted and should not be treated as accurate local metre distances for trips outside UTM zone 32N. Resolve this with the feature-generation owner before using x_m/y_m for distance-based candidates.",
        "- Raw data partitions are reported separately; student_test is descriptive only in this Task A workflow.",
        "",
        "## Missing-value and validation findings",
        "",
        f"- GPS rows across all complete trips: {int(summaries['gps_rows_original'].sum())}.",
        f"- Missing longitude values: {int(summaries['missing_lon'].sum())}; missing latitude values: {int(summaries['missing_lat'].sum())}; missing timestamps: {int(summaries['missing_timestamp'].sum())}.",
        f"- GPS rows removed for missing/non-finite fields: {int(summaries['missing_rows_removed'].sum())}; invalid-coordinate rows: {int(summaries['invalid_coordinate_rows_removed'].sum())}; exact duplicate rows: {int(summaries['duplicate_rows_removed'].sum())}.",
        f"- Invalid node coordinates: {int(summaries['invalid_node_coordinates'].sum())}; invalid arc endpoints: {int(summaries['invalid_arc_endpoints'].sum())}.",
        "",
        "## Inventory",
        "",
        *inventory_lines,
        "",
        "## Gap Findings",
        "",
        "| trip | max_sampling_gap_s | sampling_gap_count_over_2s | median_sampling_gap_s |",
        "|---|---:|---:|---:|",
    ]
    for row in max_gap_trips.itertuples(index=False):
        lines.append(
            f"| {row.trip} | {row.max_sampling_gap_s:.3f} | "
            f"{row.sampling_gap_count_over_2s} | {row.median_sampling_gap_s:.3f} |"
        )

    lines.extend(
        [
            "",
        "## Trips with Invalid, Missing, or Duplicate Points",
            "",
        ]
    )
    if missing_or_invalid.empty:
        lines.append("No trips contained invalid, missing, or duplicate GPS points.")
    else:
        lines.extend([
            "| trip | invalid_or_missing_removed | duplicate_rows_removed |",
            "|---|---:|---:|",
        ])
        for row in missing_or_invalid[
            ["trip", "invalid_or_missing_removed", "duplicate_rows_removed"]
        ].itertuples(index=False):
            lines.append(
                f"| {row.trip} | {row.invalid_or_missing_removed} | "
                f"{row.duplicate_rows_removed} |"
            )

    lines.extend(
        [
            "",
            "## Notable Anomaly",
            "",
            "The table above is data-derived. Large timestamp gaps are retained in this preprocessing stage; downstream trajectory processing should split sequences at the configured gap threshold.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--quality", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    if Transformer is None:
        transformer = _Utm32NTransformer()
    else:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    summaries: list[dict[str, float | int | str]] = []
    inventory_rows: list[dict[str, float | int | str]] = []
    gap_rows: list[dict[str, float | int | str]] = []

    for partition in ("train", "student_test"):
        partition_dir = args.dataset / partition
        if not partition_dir.exists():
            continue
        for trip_dir in trip_dirs(args.dataset, (partition,)):
            trip_id = trip_dir.name
            required = [trip_dir / f"{trip_id}.{suffix}" for suffix in ("track", "nodes", "arcs")]
            missing_files = [path.name for path in required if not path.is_file()]
            if missing_files:
                summaries.append({"trip": trip_id, "partition": partition, "valid": False, "missing_files": ", ".join(missing_files)})
                inventory_rows.append({"partition": partition, "trip": trip_id, "gps_points": 0, "nodes": 0, "arcs": 0, "missing_files": ", ".join(missing_files)})
                continue
            track, nodes, arcs, summary = preprocessing_summary(trip_dir, transformer)
            summary["partition"] = partition
            summaries.append(summary)
            gap_rows.append(summary)

            inventory_rows.append(
                {
                    "partition": partition,
                    "trip": trip_id,
                    "gps_points": len(track),
                    "nodes": len(nodes),
                    "arcs": len(arcs),
                    "track_file_bytes": (trip_dir / f"{trip_id}.track").stat().st_size,
                    "nodes_file_bytes": (trip_dir / f"{trip_id}.nodes").stat().st_size,
                    "arcs_file_bytes": (trip_dir / f"{trip_id}.arcs").stat().st_size,
                    "route_file_bytes": (
                        trip_dir / f"{trip_id}.route"
                    ).stat().st_size
                    if (trip_dir / f"{trip_id}.route").exists()
                    else 0,
                    "invalid_gps_points": summary["invalid_coordinate_rows_removed"],
                    "invalid_node_coordinates": summary["invalid_node_coordinates"],
                    "invalid_arc_endpoints": summary["invalid_arc_endpoints"],
                }
            )

    inventory = pd.DataFrame(inventory_rows)
    summaries_df = pd.DataFrame(summaries)
    gaps_df = pd.DataFrame(gap_rows)

    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.inventory, index=False)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summaries_df.to_csv(args.summary, index=False)
    write_quality_markdown(inventory, summaries_df, gaps_df, args.quality)


if __name__ == "__main__":
    main()
