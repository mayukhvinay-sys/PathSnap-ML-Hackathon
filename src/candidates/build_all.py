"""Build candidate tables for all train and student_test trips (with labels).

Run from the repo root:  python -m src.candidates.build_all
Outputs (data/processed/): candidates_train.parquet, candidates_student.parquet,
ref_arc_train.parquet, ref_arc_student.parquet.
"""
import time

import pandas as pd

from ._io import REPO_ROOT, list_trips, load_config, load_trip
from .align_labels import add_labels
from .candidate_gen import build_candidates

OUT_NAMES = {"train": "train", "student_test": "student"}


def main():
    cfg = load_config("candidates.yaml")
    out_dir = REPO_ROOT / "data" / "processed"
    for part, short in OUT_NAMES.items():
        frames, refs = [], []
        t0 = time.time()
        for tid in list_trips(part):
            trip = load_trip(tid, part, with_route=True)
            c = build_candidates(trip, cfg["radius_m"], cfg["top_k"], cfg["gap_seconds"])
            c, ref = add_labels(c, trip, cfg["label_max_dist_m"])
            # points with no reference arc have no valid label: keep them out of the table
            c = c[ref["ref_arc"].values[c["obs_idx"].values] >= 0]
            frames.append(c)
            refs.append(ref)
        cands = pd.concat(frames, ignore_index=True)
        ref_df = pd.concat(refs, ignore_index=True)
        cands.to_parquet(out_dir / f"candidates_{short}.parquet", index=False)
        ref_df.to_parquet(out_dir / f"ref_arc_{short}.parquet", index=False)
        pts = len(ref_df)
        print(f"{part}: trips={len(frames)} points={pts} rows={len(cands)} "
              f"positive_rate={cands['label'].mean():.3f} "
              f"points_without_ref={(ref_df['ref_arc'] < 0).sum()} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
