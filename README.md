# PathSnap-ML-Hackathon

GPS map-matching and route reconstruction: candidate arcs and features, classifiers, HMM/Viterbi decoding, FastAPI backend and evaluator UI.

## Setup
```
pip install -r requirements.txt
```
The raw dataset is not committed. Put it next to this folder and set `data_root` in `configs/paths.yaml`, or set the environment variable `PATHSNAP_DATA_ROOT` to the folder that contains `train/` and `student_test/`.

## Build the candidate tables (about 30 s)
```
python -m src.candidates.build_all
```
Creates `data/processed/candidates_train.parquet`, `candidates_student.parquet` and `ref_arc_*.parquet`. Features are listed in `configs/features.yaml`; see `results/features.md`.

## Run the API (Laptop 1)
From the repo root:
```
set PATHSNAP_TRIP_DIR=<folder containing one sub-folder per trip>
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Defaults: model `artifacts/models/RandomForest.joblib` (override with `PATHSNAP_MODEL`), HMM settings `artifacts/hmm_params.json`, candidate settings `configs/candidates.yaml`. If `PATHSNAP_TRIP_DIR` is not set, the `student_test` partition is used.

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, model name, number of trips |
| `GET /trips` | list of trip ids |
| `POST /predict` with `{"trip_id": "..."}` | predicted route, per-point arcs and metrics |

The prediction path never reads the `.route` file. Accuracy is computed server-side in `src/api/evaluator.py`, and the response contains no reference route or labels. Every request is logged to the console.

Check from another laptop: `curl http://<Laptop-1-IP>:8000/health`. Allow inbound TCP port 8000 in the firewall of Laptop 1.

## Tests
```
python -m tests.test_no_leakage
python -m tests.test_api
```

## Layout
- `configs/` shared settings and column contracts
- `src/data`, `src/candidates`, `src/models`, `src/tuning`, `src/decoder`, `src/viz`, `src/api`
- `ui/` evaluator interface
- `artifacts/` saved models and decoder parameters
- `results/` tables, figures, maps
- `tests/` checks, including the no-leakage test
