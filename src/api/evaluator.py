"""Server-side evaluation. The only place that reads the reference route.

The reference route is used to compute accuracy and is never returned to the UI.
"""
import numpy as np

from src.candidates.align_labels import align_points_to_route
from src.decoder.continuity import route_continuity

from .config import CANDIDATE_CFG
from .trip_loader import load_trip_from_dir


def evaluate(trip_dir, trip_id, pred):
    """Top-1 accuracy (classifier only and with HMM) and route continuity.

    Returns None values if the trip has no reference route available.
    """
    empty = {"top1_acc_classifier": None, "top1_acc_hmm": None,
             "route_continuity": None, "route_continuity_classifier": None,
             "n_points_scored": 0}
    trip = load_trip_from_dir(trip_dir, trip_id, with_route=True)
    if trip.route is None or len(trip.route) == 0:
        return empty
    ref = align_points_to_route(trip.track, trip.nodes, trip.arcs, trip.route,
                                CANDIDATE_CFG["label_max_dist_m"])
    mask = ref >= 0
    if not mask.any():
        return empty
    return {
        "top1_acc_classifier": float(np.mean(pred["clf_arcs"][mask] == ref[mask])),
        "top1_acc_hmm": float(np.mean(pred["hmm_arcs"][mask] == ref[mask])),
        "route_continuity": float(route_continuity(pred["hmm_arcs"], trip.arcs)),
        "route_continuity_classifier": float(route_continuity(pred["clf_arcs"], trip.arcs)),
        "n_points_scored": int(mask.sum()),
    }
