"""Dummy candidate tables with the agreed schema (contract A), so Member C can run everything
before Member B's real files (H1/H2) exist.   Usage:
    python tests/make_dummy_candidates.py            # writes data/processed/dummy/*.parquet
    PATHSNAP_DATA=dummy python -m src.models.run_phase1       (Windows: set PATHSNAP_DATA=dummy)
The data is random but learnable (true arc is usually closer / better aligned); it is NOT real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_partition(n_trips: int, id_base: int, seed: int, miss_rate: float = 0.05):
    rng = np.random.default_rng(seed)
    rows, refs = [], []
    for i in range(n_trips):
        tid = f"{id_base + i:08d}"
        n_pts = int(rng.integers(40, 90))
        arc = 1000 * i
        x = y = 0.0
        t = 0
        for obs in range(n_pts):
            gap = obs > 0 and rng.random() < 0.03
            t += int(rng.integers(40, 200)) if gap else 1
            if obs and obs % 4 == 0:
                arc += 1
            x += rng.normal(8, 2); y += rng.normal(0, 2)
            speed = max(0.0, rng.normal(10, 3))
            n_c = int(rng.integers(3, 11))
            hit = rng.random() > miss_rate
            arcs = list(rng.choice(np.arange(5000, 9000), size=n_c, replace=False))
            dists = list(rng.uniform(5, 80, size=n_c))
            heads = list(rng.uniform(0, 180, size=n_c))
            true_idx = int(rng.integers(0, n_c)) if hit else -1
            if hit:
                arcs[true_idx] = arc
                dists[true_idx] = abs(rng.normal(0, 9)) + 1
                heads[true_idx] = abs(rng.normal(0, 15))
            order = np.argsort(dists)
            best = min(dists)
            for rank, j in enumerate(order):
                rows.append({
                    "trip_id": tid, "obs_idx": obs, "t": t, "x_m": x, "y_m": y,
                    "lon": 9.0 + x / 70000, "lat": 52.0 + y / 111000,
                    "arc_id": int(arcs[j]), "after_gap": int(gap),
                    "dist_m": float(dists[j]), "offset_frac": float(rng.random()),
                    "arc_len_m": float(rng.lognormal(4, 0.6)), "heading_diff_deg": float(heads[j]),
                    "speed_mps": float(speed), "rank_dist": rank + 1,
                    "dist_gap_best": float(dists[j] - best), "n_cands": n_c,
                    "has_twin": int(rng.random() < 0.3),
                    "label": int(hit and j == true_idx),
                })
            refs.append({"trip_id": tid, "obs_idx": obs, "ref_arc": arc})  # on a miss, arc is not among the candidates
    return pd.DataFrame(rows), pd.DataFrame(refs)


def main():
    out = ROOT / "data" / "processed" / "dummy"
    out.mkdir(parents=True, exist_ok=True)
    for part, n, base, seed in (("train", 24, 10000000, 1), ("student", 6, 20000000, 2)):
        c, r = make_partition(n, base, seed)
        c.to_parquet(out / f"candidates_{part}.parquet", index=False)
        r.to_parquet(out / f"ref_arc_{part}.parquet", index=False)
        print(f"{part}: {len(c):,} candidate rows, {c.trip_id.nunique()} trips, "
              f"{int(c.groupby(['trip_id','obs_idx']).label.max().eq(0).sum())} points without a hit")
    print("written to", out)


if __name__ == "__main__":
    sys.exit(main())
