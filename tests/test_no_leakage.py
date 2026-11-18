"""Leakage checks: candidate generation must not depend on the reference route.

Run from the repo root:  python -m tests.test_no_leakage
(also works with pytest)
"""
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.candidates import _io
from src.candidates.align_labels import add_labels
from src.candidates.candidate_gen import build_candidates

TRIP = "00000005"
PART = "train"


def _copy_trip(dst_root, with_route):
    src = _io.data_root() / PART / TRIP
    dst = Path(dst_root) / PART / TRIP
    dst.mkdir(parents=True)
    exts = [".track", ".nodes", ".arcs"] + ([".route"] if with_route else [])
    for ext in exts:
        shutil.copy(src / f"{TRIP}{ext}", dst / f"{TRIP}{ext}")


def _load_from(root, with_route):
    original = _io.data_root
    _io.data_root = lambda: Path(root)
    try:
        return _io.load_trip(TRIP, PART, with_route=with_route)
    finally:
        _io.data_root = original


def test_candidates_work_without_route_file():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_trip(tmp, with_route=False)
        assert not (Path(tmp) / PART / TRIP / f"{TRIP}.route").exists()
        trip = _load_from(tmp, with_route=False)
        assert trip.route is None
        c = build_candidates(trip)
        assert len(c) > 0
        assert "label" not in c.columns


def test_candidates_identical_with_or_without_route():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _copy_trip(a, with_route=False)
        _copy_trip(b, with_route=True)
        c_without = build_candidates(_load_from(a, with_route=False))
        c_with = build_candidates(_load_from(b, with_route=True))
        pd.testing.assert_frame_equal(c_without, c_with)


def test_labels_require_route():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_trip(tmp, with_route=False)
        trip = _load_from(tmp, with_route=False)
        c = build_candidates(trip)
        try:
            add_labels(c, trip)
        except ValueError:
            return
        raise AssertionError("add_labels must refuse to run without a route")


if __name__ == "__main__":
    test_candidates_work_without_route_file()
    test_candidates_identical_with_or_without_route()
    test_labels_require_route()
    print("all leakage tests passed")
