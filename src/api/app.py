"""FastAPI backend (runs on Laptop 1).

Start from the repo root:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Endpoints: GET /health, GET /trips, POST /predict {"trip_id": "..."}
The response never contains the reference route or labels.
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .evaluator import evaluate
from .predictor import Predictor
from .trip_loader import list_trip_ids, load_trip_from_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")
state = {}


@asynccontextmanager
async def lifespan(app):
    state["predictor"] = Predictor()
    log.info("model loaded: %s | trip folder: %s | trips found: %d",
             state["predictor"].model_name, config.TRIP_DIR, len(list_trip_ids(config.TRIP_DIR)))
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _path_from_arcs(trip, arcs):
    """Polyline [[lat, lon], ...] following the predicted arcs in order."""
    lat, lon = trip.nodes["lat"].values, trip.nodes["lon"].values
    path, last_to = [], None
    prev = None
    for a in arcs:
        if a < 0 or a == prev:
            continue
        prev = a
        f, t = int(trip.arcs[a, 0]), int(trip.arcs[a, 1])
        if last_to != f:
            path.append([float(lat[f]), float(lon[f])])
        path.append([float(lat[t]), float(lon[t])])
        last_to = t
    return path


@app.get("/health")
def health():
    return {"status": "ok", "model": state["predictor"].model_name,
            "n_trips": len(list_trip_ids(config.TRIP_DIR))}


@app.get("/trips")
def trips():
    return {"trips": list_trip_ids(config.TRIP_DIR)}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    log.info("POST /predict trip_id=%s", req.trip_id)
    try:
        trip = load_trip_from_dir(config.TRIP_DIR, req.trip_id, with_route=False)
    except KeyError:
        log.warning("unknown trip_id=%s", req.trip_id)
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id}")
    pred = state["predictor"].predict(trip)
    metrics = evaluate(config.TRIP_DIR, req.trip_id, pred)       # server-side only
    metrics.update({"latency_s": round(time.perf_counter() - t0, 3),
                    "n_points": int(len(trip.track))})
    log.info("trip_id=%s points=%d top1_clf=%s top1_hmm=%s continuity=%s latency=%.2fs",
             req.trip_id, len(trip.track), metrics["top1_acc_classifier"],
             metrics["top1_acc_hmm"], metrics["route_continuity"], metrics["latency_s"])
    return {
        "trip_id": req.trip_id,
        "gps": trip.track[["lat", "lon"]].values.tolist(),
        "pred_arcs": pred["hmm_arcs"].tolist(),
        "pred_path": _path_from_arcs(trip, pred["hmm_arcs"]),
        "classifier_path": _path_from_arcs(trip, pred["clf_arcs"]),
        "metrics": metrics,
    }
