"""API tests. Run from the repo root:  python -m tests.test_api   (also works with pytest)

Uses the trip folder from src/api/config.py (set PATHSNAP_TRIP_DIR or PATHSNAP_DATA_ROOT
if your data is elsewhere).
"""
from fastapi.testclient import TestClient

from src.api import config
from src.api.app import app
from src.api.trip_loader import list_trip_ids

FORBIDDEN_KEYS = {"route", "ref_arc", "reference", "label", "labels", "ground_truth"}


def _smallest_trip():
    ids = list_trip_ids(config.TRIP_DIR)
    if not ids:
        return None
    sizes = {t: sum(1 for _ in open(config.TRIP_DIR / t / f"{t}.track")) for t in ids}
    return min(sizes, key=sizes.get)


def test_health_and_trips():
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok"
        assert isinstance(c.get("/trips").json()["trips"], list)


def test_predict_response_shape_and_no_ground_truth():
    trip = _smallest_trip()
    if trip is None:
        print("skip: no trips found in", config.TRIP_DIR)
        return
    with TestClient(app) as c:
        r = c.post("/predict", json={"trip_id": trip})
        assert r.status_code == 200
        j = r.json()
        for key in ("trip_id", "gps", "pred_arcs", "pred_path", "metrics"):
            assert key in j
        assert len(j["gps"]) == len(j["pred_arcs"]) == j["metrics"]["n_points"]
        assert not (FORBIDDEN_KEYS & set(j)), "response must not expose reference data"
        assert 0.0 <= j["metrics"]["top1_acc_hmm"] <= 1.0


def test_unknown_and_malicious_trip_ids():
    with TestClient(app) as c:
        assert c.post("/predict", json={"trip_id": "99999999"}).status_code == 404
        assert c.post("/predict", json={"trip_id": "../train/00000001"}).status_code == 404
        assert c.post("/predict", json={}).status_code == 422


if __name__ == "__main__":
    test_health_and_trips()
    test_predict_response_shape_and_no_ground_truth()
    test_unknown_and_malicious_trip_ids()
    print("all API tests passed")
