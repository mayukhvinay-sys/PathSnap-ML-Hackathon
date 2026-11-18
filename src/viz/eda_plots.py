from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_arcs, load_nodes, load_track, trip_dirs
from data.preprocess import preprocessing_summary


def haversine_km(lon: np.ndarray, lat: np.ndarray) -> float:
    lon = np.radians(lon)
    lat = np.radians(lat)
    if len(lon) < 2:
        return 0.0
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    return float(6371.0 * np.sum(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partition", choices=("train", "student_test", "all"), default="all")
    parser.add_argument("--sample-trip", default="00000001")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    args.output.mkdir(parents=True, exist_ok=True)

    tracks: list[pd.DataFrame] = []
    intervals: list[float] = []
    lengths: list[dict[str, float | str]] = []
    network_stats: list[dict[str, float | int | str]] = []

    partitions = ("train", "student_test") if args.partition == "all" else (args.partition,)
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    except ImportError:
        from data.preprocess import _Utm32NTransformer
        transformer = _Utm32NTransformer()

    for partition in partitions:
        for trip_dir in trip_dirs(args.dataset, (partition,)):
            trip_id = trip_dir.name
            track, _, _, _ = preprocessing_summary(trip_dir, transformer)
            track["trip"] = trip_id
            track["partition"] = partition
            tracks.append(track)

            sorted_track = track.sort_values("timestamp")
            intervals.extend(sorted_track["timestamp"].diff().dropna().tolist())
            lengths.append(
                {
                    "partition": partition,
                    "trip": trip_id,
                    "trajectory_length_km": haversine_km(
                        sorted_track["lon"].to_numpy(),
                        sorted_track["lat"].to_numpy(),
                    ),
                }
            )

            nodes = load_nodes(trip_dir / f"{trip_id}.nodes")
            arcs = load_arcs(trip_dir / f"{trip_id}.arcs")
            graph = nx.Graph()
            graph.add_nodes_from(range(len(nodes)))
            graph.add_edges_from(arcs[["from_node", "to_node"]].itertuples(index=False, name=None))
            components = list(nx.connected_components(graph))
            isolated = list(nx.isolates(graph))
            network_stats.append(
                {
                    "partition": partition,
                    "trip": trip_id,
                    "nodes": len(nodes),
                    "arcs": len(arcs),
                    "components": len(components),
                    "largest_component": max((len(c) for c in components), default=0),
                    "isolated_nodes": len(isolated),
                }
            )

    all_tracks = pd.concat(tracks, ignore_index=True)
    length_df = pd.DataFrame(lengths)
    network_df = pd.DataFrame(network_stats)
    intervals_series = pd.Series(intervals, name="sampling_interval_s")

    # Persist trajectory coordinates as a proportionate visual sample, labelled by partition.
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=all_tracks.sample(min(len(all_tracks), 30000), random_state=42),
        x="lon",
        y="lat",
        hue="partition",
        s=8,
        alpha=0.45,
        ax=ax,
    )
    ax.set_title("GPS Coordinate Extent")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(args.output / "gps_extent.png", dpi=180)
    plt.close(fig)

    # Plot one cleaned trajectory against its source road network per partition.
    for partition in partitions:
        candidate = args.dataset / partition / args.sample_trip
        if not candidate.is_dir():
            available = trip_dirs(args.dataset, (partition,))
            if not available:
                continue
            candidate = available[0]
        trip_id = candidate.name
        track, nodes, arcs, _ = preprocessing_summary(candidate, transformer)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(track["lon"], track["lat"], color="#2563eb", linewidth=1.4, marker=".", markersize=3, label="Clean GPS")
        valid_arcs = arcs[arcs["from_node"].between(0, len(nodes)-1) & arcs["to_node"].between(0, len(nodes)-1)]
        # Bound dense networks for a legible overview.
        for edge in valid_arcs.iloc[::max(1, len(valid_arcs)//12000)][["from_node", "to_node"]].itertuples(index=False, name=None):
            a, b = nodes.iloc[edge[0]], nodes.iloc[edge[1]]
            ax.plot([a.lon, b.lon], [a.lat, b.lat], color="#94a3b8", alpha=.24, linewidth=.45)
        ax.set_title(f"Clean GPS trajectory and road network — {partition} trip {trip_id}")
        ax.set_xlabel("Longitude (WGS84)")
        ax.set_ylabel("Latitude (WGS84)")
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(args.output / f"trajectory_network_{partition}_{trip_id}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(intervals_series[intervals_series <= 100], bins=100, color="#4C72B0")
    ax.set_title("GPS Sampling Intervals")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(args.output / "sampling_intervals.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(length_df["trajectory_length_km"], bins=30, ax=ax)
    ax.set_title("Trajectory Length Distribution")
    ax.set_xlabel("Length (km)")
    ax.set_ylabel("Trip count")
    fig.tight_layout()
    fig.savefig(args.output / "trajectory_lengths.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=network_df,
        x="nodes",
        y="arcs",
        hue="partition",
        ax=ax,
    )
    ax.set_title("Road Network Size")
    ax.set_xlabel("Node count")
    ax.set_ylabel("Arc count")
    fig.tight_layout()
    fig.savefig(args.output / "network_size.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(network_df["components"], bins=30, ax=ax)
    ax.set_title("Network Connected-Component Count")
    ax.set_xlabel("Components")
    ax.set_ylabel("Trip count")
    fig.tight_layout()
    fig.savefig(args.output / "network_components.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(network_df["isolated_nodes"], bins=30, ax=ax)
    ax.set_title("Isolated Network Nodes")
    ax.set_xlabel("Isolated node count")
    ax.set_ylabel("Trip count")
    fig.tight_layout()
    fig.savefig(args.output / "isolated_nodes.png", dpi=180)
    plt.close(fig)

    length_df.to_csv(args.output / "trajectory_lengths.csv", index=False)
    network_df.to_csv(args.output / "network_connectivity.csv", index=False)
    pd.DataFrame({"partition": [args.partition], "sample_trip_requested": [args.sample_trip]}).to_csv(args.output / "analysis_scope.csv", index=False)


if __name__ == "__main__":
    main()
