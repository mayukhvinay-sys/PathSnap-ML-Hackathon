"""Viterbi decoder for map-matching candidate arcs."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _arc_endpoints(trip: Any) -> np.ndarray:
    arcs = np.asarray(trip.arcs)
    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError("trip.arcs must be an (n_arcs, 2) array of start/end node ids")
    return arcs


def _hop_distances(arcs: np.ndarray, max_hops: int) -> list[dict[int, int]]:
    """For each arc, find reachable arc-start nodes from its end node."""
    outgoing: dict[int, list[int]] = defaultdict(list)
    for arc_id, (start, _end) in enumerate(arcs):
        outgoing[int(start)].append(arc_id)

    all_distances: list[dict[int, int]] = []
    for _arc_id, (_start, end) in enumerate(arcs):
        distances: dict[int, int] = {}
        queue = deque([(int(end), 0)])
        seen = {int(end)}
        while queue:
            node, hops = queue.popleft()
            if hops >= max_hops:
                continue
            for next_arc in outgoing.get(node, ()):
                next_start, next_end = map(int, arcs[next_arc])
                distances[next_arc] = min(distances.get(next_arc, max_hops + 1), hops)
                if next_end not in seen:
                    seen.add(next_end)
                    queue.append((next_end, hops + 1))
        all_distances.append(distances)
    return all_distances


def _emission_values(cands: pd.DataFrame, p: Any) -> np.ndarray:
    """Convert candidate scores to per-observation log probabilities."""
    if len(cands) == 0:
        return np.empty(0, dtype=float)
    if p is None:
        if "p" in cands:
            values = cands["p"].to_numpy(dtype=float)
        elif "dist_m" in cands:
            values = 1.0 / (1.0 + cands["dist_m"].to_numpy(dtype=float))
        else:
            raise ValueError("Pass candidate scores p, or provide cands['p'] / cands['dist_m']")
    elif isinstance(p, pd.Series):
        values = p.to_numpy(dtype=float)
    else:
        values = np.asarray(p, dtype=float).reshape(-1)
        if len(values) != len(cands):
            raise ValueError("p must contain one score for each row of cands")
    if len(values) != len(cands):
        raise ValueError("p must contain one score for each row of cands")

    # Accept probabilities or positive unnormalized scores. Clamp invalid/zero
    # values so they cannot make the entire path impossible.
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 1e-12, None)
    logs = np.log(values)
    for _obs_idx, positions in cands.groupby("obs_idx", sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        row = logs[positions]
        row -= np.logaddexp.reduce(row)
        logs[positions] = row
    return logs


def decode_trip(
    cands: pd.DataFrame,
    p: Any,
    trip: Any,
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Decode the most likely road-arc sequence for a trip.

    Args:
        cands: Candidate rows with ``obs_idx`` and ``arc_id``. ``after_gap`` is
            optional and marks a sequence reset before that observation.
        p: One positive score/probability per candidate row. ``None`` uses the
            ``p`` or ``dist_m`` column as a convenient baseline.
        trip: Object with ``arcs``, an (n_arcs, 2) start/end node array, and
            optionally ``track`` (used to include points that have no candidates).
        params: Optional ``beta`` transition penalty, ``max_route_hops`` and
            ``max_gap_s``. A larger beta more strongly favors connected roads.

    Returns:
        DataFrame with exactly one row per GPS observation: ``obs_idx, arc_id``.
        ``arc_id`` is -1 when that observation has no candidates.
    """
    params = dict(params or {})
    beta = float(params.get("beta", 2.0))
    max_hops = max(1, int(params.get("max_route_hops", 6)))
    disconnected_penalty = float(params.get("disconnected_penalty", 8.0))
    if beta < 0 or disconnected_penalty < 0:
        raise ValueError("beta and disconnected_penalty must be non-negative")
    cands = cands.reset_index(drop=True).copy()
    if cands.empty:
        n_obs = len(trip.track) if hasattr(trip, "track") else 0
        return pd.DataFrame({"obs_idx": np.arange(n_obs, dtype=int),
                             "arc_id": np.full(n_obs, -1, dtype=int)})
    if not {"obs_idx", "arc_id"}.issubset(cands.columns):
        raise ValueError("cands must contain obs_idx and arc_id columns")

    arcs = _arc_endpoints(trip)
    log_emissions = _emission_values(cands, p)
    hop_distances = _hop_distances(arcs, max_hops)
    n_obs = len(trip.track) if hasattr(trip, "track") else int(cands["obs_idx"].max()) + 1
    obs_to_rows = {int(k): np.asarray(v, dtype=int) for k, v in cands.groupby("obs_idx", sort=False).indices.items()}

    decoded = np.full(n_obs, -1, dtype=int)
    segment_states: list[np.ndarray] = []
    segment_backs: list[np.ndarray | None] = []
    previous_scores: np.ndarray | None = None

    def finish_segment() -> None:
        nonlocal segment_states, segment_backs, previous_scores
        if not segment_states:
            return
        state_index = int(np.argmax(previous_scores))
        for layer in range(len(segment_states) - 1, -1, -1):
            decoded[segment_obs[layer]] = int(segment_states[layer][state_index])
            back = segment_backs[layer]
            if back is not None:
                state_index = int(back[state_index])
        segment_states = []
        segment_backs = []
        previous_scores = None

    segment_obs: list[int] = []
    for obs_idx in range(n_obs):
        rows = obs_to_rows.get(obs_idx)
        if rows is None or len(rows) == 0:
            finish_segment()
            segment_obs = []
            continue

        states = cands.loc[rows, "arc_id"].to_numpy(dtype=int)
        if np.any(states < 0) or np.any(states >= len(arcs)):
            raise ValueError(f"candidate arc_id out of range at obs_idx={obs_idx}")
        is_gap = bool(cands.loc[rows[0], "after_gap"]) if "after_gap" in cands else False
        if is_gap and segment_states:
            finish_segment()
            segment_obs = []
        if not segment_states:
            scores = log_emissions[rows].copy()
            back = None
        else:
            previous_states = segment_states[-1]
            transition_scores = np.full((len(previous_states), len(states)), -disconnected_penalty, dtype=float)
            for i, prev_arc in enumerate(previous_states):
                reachable = hop_distances[int(prev_arc)]
                for j, next_arc in enumerate(states):
                    next_arc = int(next_arc)
                    if next_arc == int(prev_arc):
                        transition_scores[i, j] = 0.0
                    elif next_arc in reachable:
                        # Zero additional hops means directly connected arcs;
                        # additional hops allow sparse GPS sampling.
                        transition_scores[i, j] = -beta * reachable[next_arc]
            totals = previous_scores[:, None] + transition_scores
            back = np.argmax(totals, axis=0)
            scores = totals[back, np.arange(len(states))] + log_emissions[rows]

        segment_states.append(states)
        segment_backs.append(back)
        segment_obs.append(obs_idx)
        previous_scores = scores

    finish_segment()
    return pd.DataFrame({"obs_idx": np.arange(n_obs, dtype=int), "arc_id": decoded})
