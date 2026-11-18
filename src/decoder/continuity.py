"""Route continuity metric for decoded arc sequences."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def route_continuity(pred_arcs: Any, trip_arcs: Any) -> float:
    """Return the fraction of consecutive valid predictions connected in order.

    ``pred_arcs`` may be a sequence or a DataFrame with an ``arc_id`` column.
    Missing predictions (negative arc ids) are excluded from the denominator.
    A transition counts as connected when it stays on the same arc or the end
    node of the first directed arc is the start node of the next.
    """
    if isinstance(pred_arcs, pd.DataFrame):
        pred_arcs = pred_arcs["arc_id"].to_numpy()
    pred = np.asarray(pred_arcs, dtype=int).reshape(-1)
    arcs = np.asarray(trip_arcs)
    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError("trip_arcs must be an (n_arcs, 2) start/end node array")

    pairs = [(int(a), int(b)) for a, b in zip(pred[:-1], pred[1:]) if a >= 0 and b >= 0]
    if not pairs:
        return float("nan")
    connected = 0
    for a, b in pairs:
        if a >= len(arcs) or b >= len(arcs):
            raise ValueError("predicted arc_id is outside trip_arcs")
        connected += int(a == b or arcs[a, 1] == arcs[b, 0])
    return connected / len(pairs)
