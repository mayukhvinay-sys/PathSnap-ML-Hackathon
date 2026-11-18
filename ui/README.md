# PathSnap Evaluator Interface

## Run

Install the UI dependencies from the repository root:

```bash
python -m pip install -r ui/requirements.txt
```

Set the API base URL if the server is on another machine, then start the interface:

```bash
# Windows PowerShell
$env:PATHSNAP_API_URL = "http://127.0.0.1:8000"
streamlit run ui/app.py
```

For a server on another laptop, replace the address with that laptop's reachable IP, for example `http://192.168.1.20:8000`. The UI also has an API URL field in the sidebar.

Turn on **Use demo response** in the sidebar to preview the layout with `mock_predict_response.json` before the API is available. Demo output is explicitly marked as mock data.

## API expectations

- `GET /trips` returns `{ "trips": ["00000000", ...] }`.
- `POST /predict` accepts `{ "trip_id": "00000000" }` and returns `trip_id`, `gps`, `pred_arcs`, `pred_path`, `classifier_path`, and `metrics`. GPS and route coordinates are `[latitude, longitude]` pairs. The UI shows classifier-only and HMM/Viterbi paths as separate layers and displays classifier/HMM top-1 scores, route continuity, point count, and latency. Metric values can be `null` when no reference route is available; those display as unavailable.

The interface does not request or display reference routes or labels. It draws GPS observations and the predicted path as separate map layers.
