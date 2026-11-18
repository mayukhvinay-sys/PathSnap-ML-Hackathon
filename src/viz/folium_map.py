from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

try:
    import folium
    from folium import plugins
except ImportError as exc:  # pragma: no cover - dependency declared in requirements
    raise ImportError("Folium is required to create trip maps; install requirements.txt") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import Trip, list_trips, load_trip


def _arc_segments(trip: Trip, arc_ids) -> list[list[list[float]]]:
    segments = []
    for arc_id in np.asarray(arc_ids, dtype=np.int64).ravel():
        if 0 <= arc_id < len(trip.arcs):
            start, end = trip.arcs[arc_id]
            segments.append([
                [float(trip.nodes.iloc[start]["lat"]), float(trip.nodes.iloc[start]["lon"])],
                [float(trip.nodes.iloc[end]["lat"]), float(trip.nodes.iloc[end]["lon"])],
            ])
    return segments


def make_map(
    trip: Trip,
    pred_arcs=None,
    ref_route=None,
    out_html: str | Path = "map.html",
) -> Path:
    """Create a layered map of GPS, road network and optional reference/predicted routes."""
    if trip.track.empty:
        raise ValueError(f"Trip {trip.trip_id} has no GPS observations to map")
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    center = [float(trip.track["lat"].mean()), float(trip.track["lon"].mean())]
    map_object = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    locations = trip.track[["lat", "lon"]].to_numpy(dtype=float).tolist()
    if len(locations) > 2000:
        plugins.FastMarkerCluster(locations, name="GPS observations").add_to(gps_layer)
    else:
        for lat, lon in locations:
            folium.CircleMarker([lat, lon], radius=2, color="#2563eb", fill=True,
                                fill_color="#2563eb", fill_opacity=.75, weight=1).add_to(gps_layer)
    if len(locations) > 1:
        folium.PolyLine(locations, color="#60a5fa", weight=1.5, opacity=.6,
                        tooltip="Observed GPS sequence").add_to(gps_layer)
    gps_layer.add_to(map_object)

    network_layer = folium.FeatureGroup(name="Road network", show=False)
    network_step = max(1, int(np.ceil(len(trip.arcs) / 20000)))
    for start, end in trip.arcs[::network_step]:
        folium.PolyLine(
            [[trip.nodes.iloc[start]["lat"], trip.nodes.iloc[start]["lon"]],
             [trip.nodes.iloc[end]["lat"], trip.nodes.iloc[end]["lon"]]],
            color="#94a3b8", weight=1, opacity=.45,
        ).add_to(network_layer)
    network_layer.add_to(map_object)

    if ref_route is None:
        ref_route = trip.route
    if ref_route is not None:
        reference_layer = folium.FeatureGroup(name="Reference route", show=True)
        for segment in _arc_segments(trip, ref_route):
            folium.PolyLine(segment, color="#16a34a", weight=5, opacity=.9).add_to(reference_layer)
        reference_layer.add_to(map_object)

    if pred_arcs is not None:
        predicted_layer = folium.FeatureGroup(name="Predicted route", show=True)
        for segment in _arc_segments(trip, pred_arcs):
            folium.PolyLine(segment, color="#f59e0b", weight=4, opacity=.9).add_to(predicted_layer)
        predicted_layer.add_to(map_object)

    legend_rows = [
        ("#2563eb", "●", "GPS observations"),
        ("#94a3b8", "━", "Road network (sampled for display)" if network_step > 1 else "Road network"),
    ]
    if ref_route is not None:
        legend_rows.append(("#16a34a", "━", "Reference route"))
    if pred_arcs is not None:
        legend_rows.append(("#f59e0b", "━", "Predicted route"))
    legend = "<b>Trip map layers</b><br>" + "<br>".join(
        f'<span style="color:{color};font-weight:bold">{glyph}</span> {label}'
        for color, glyph, label in legend_rows
    )
    folium.Element(
        f'<div style="position:fixed;bottom:28px;left:28px;z-index:9999;'
        f'background:white;padding:10px 13px;border:1px solid #cbd5e1;'
        f'border-radius:8px;font:13px/1.6 Arial;box-shadow:0 1px 6px #64748b">{legend}</div>'
    ).add_to(map_object.get_root().html)
    folium.LayerControl(collapsed=False).add_to(map_object)
    plugins.Fullscreen(position="topright").add_to(map_object)

    bounds = np.asarray(locations)
    map_object.fit_bounds([[bounds[:, 0].min(), bounds[:, 1].min()],
                           [bounds[:, 0].max(), bounds[:, 1].max()]])
    map_object.save(str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--trip", default="00000001")
    parser.add_argument("--partition", choices=("train", "student_test"), default="train")
    parser.add_argument("--output", type=Path, default=Path("results/maps/trip.html"))
    args = parser.parse_args()
    if args.trip not in list_trips(args.partition, args.dataset):
        raise FileNotFoundError(f"Trip {args.trip} not found in {args.partition}")
    trip = load_trip(args.trip, args.partition, dataset_root=args.dataset)
    make_map(trip, out_html=args.output)


if __name__ == "__main__":
    main()
