from __future__ import annotations

import json
import os
from pathlib import Path

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parents[1]
MOCK_PATH = Path(__file__).with_name("mock_predict_response.json")
DEFAULT_API = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PathSnap Route Viewer", page_icon="🧭", layout="wide")
st.markdown("""
<style>
  .stApp { background: #f5f7fb; }
  [data-testid="stHeader"] { background: rgba(245,247,251,.85); }
  .hero { padding: 1.15rem 1.5rem; border-radius: 18px; color: white;
    background: linear-gradient(115deg,#102a43,#176b87 65%,#36a3a5); margin-bottom: 1rem; }
  .hero h1 { margin: 0; font-size: 2rem; }
  .hero p { margin: .35rem 0 0; opacity: .86; }
  [data-testid="stMetric"] { background:white; border:1px solid #e2e8f0;
    padding: .8rem 1rem; border-radius: 14px; box-shadow: 0 2px 8px #0f172a0a; }
</style>
<div class="hero"><h1>PathSnap</h1><p>GPS trajectory map matching · Route evaluation</p></div>
""", unsafe_allow_html=True)


def fetch_trips(api_url: str) -> list[str]:
    response = requests.get(f"{api_url}/trips", timeout=8)
    response.raise_for_status()
    payload = response.json()
    trips = payload.get("trips")
    if not isinstance(trips, list) or not all(isinstance(t, str) for t in trips):
        raise ValueError("GET /trips must return {\"trips\": [\"trip_id\", ...]}")
    return trips


def request_prediction(api_url: str, trip_id: str) -> dict:
    response = requests.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=90)
    response.raise_for_status()
    payload = response.json()
    required = {"trip_id", "gps", "pred_arcs", "pred_path", "classifier_path", "metrics"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Prediction response is missing: {', '.join(sorted(missing))}")
    return payload


def mock_prediction() -> dict:
    return json.loads(MOCK_PATH.read_text(encoding="utf-8"))


def _pct(value):
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _seconds(value):
    if value is None:
        return "—"
    try:
        return f"{float(value):.3f} s"
    except (TypeError, ValueError):
        return "—"


def build_route_map(payload: dict) -> folium.Map:
    gps = payload.get("gps") or []
    path = payload.get("pred_path") or []
    classifier_path = payload.get("classifier_path") or []
    coords = [p for p in gps + path + classifier_path if isinstance(p, (list, tuple)) and len(p) == 2]
    if not coords:
        raise ValueError("This response has no GPS or predicted coordinates to map")
    center = [sum(p[0] for p in coords) / len(coords), sum(p[1] for p in coords) / len(coords)]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    for point in gps:
        folium.CircleMarker(point, radius=3, color="#2563eb", fill=True,
                            fill_color="#2563eb", fill_opacity=.75, weight=1).add_to(gps_layer)
    if len(gps) > 1:
        folium.PolyLine(gps, color="#60a5fa", weight=2, opacity=.7).add_to(gps_layer)
    gps_layer.add_to(m)
    if path:
        predicted = folium.FeatureGroup(name="HMM/Viterbi route", show=True)
        folium.PolyLine(path, color="#f59e0b", weight=5, opacity=.95,
                        tooltip="HMM/Viterbi predicted route").add_to(predicted)
        folium.Marker(path[0], tooltip="Predicted start", icon=folium.Icon(color="green", icon="play")).add_to(predicted)
        folium.Marker(path[-1], tooltip="Predicted end", icon=folium.Icon(color="red", icon="stop")).add_to(predicted)
        predicted.add_to(m)
    if classifier_path:
        classifier = folium.FeatureGroup(name="Classifier-only route", show=False)
        folium.PolyLine(classifier_path, color="#8b5cf6", weight=4, opacity=.9,
                        tooltip="Classifier-only route").add_to(classifier)
        classifier.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[min(p[0] for p in coords), min(p[1] for p in coords)],
                  [max(p[0] for p in coords), max(p[1] for p in coords)]])
    return m


with st.sidebar:
    st.subheader("Connection")
    api_url = st.text_input("API base URL", value=DEFAULT_API, help="Example: http://192.168.1.20:8000")
    demo_mode = st.toggle("Use demo response", value=False, help="Preview the interface with the bundled mock response")

if demo_mode:
    mock = mock_prediction()
    trip_ids = [mock["trip_id"]]
    st.info("Demo mode is on. The map and metrics below use mock data, not a live prediction.")
else:
    try:
        trip_ids = fetch_trips(api_url.rstrip("/"))
        if not trip_ids:
            st.warning("The API returned no trips.")
    except requests.RequestException as exc:
        trip_ids = []
        st.error(f"Could not reach the PathSnap API at {api_url}. Check the address and that the server is running. Details: {exc}")
    except (ValueError, json.JSONDecodeError) as exc:
        trip_ids = []
        st.error(f"The API returned an invalid trip list: {exc}")

if trip_ids:
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    with left:
        selected_trip = st.selectbox("Evaluation trip", trip_ids, key="trip_select")
    with right:
        predict = st.button("Reconstruct route", type="primary", use_container_width=True, disabled=demo_mode)

    if demo_mode:
        st.session_state["prediction"] = mock
    elif predict:
        try:
            with st.spinner(f"Matching GPS observations for trip {selected_trip}…"):
                st.session_state["prediction"] = request_prediction(api_url.rstrip("/"), selected_trip)
            st.success("Route prediction received.")
        except requests.Timeout:
            st.error("The prediction request timed out. Try again or check the server logs.")
        except requests.HTTPError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            st.error(f"The API rejected the request: {detail}")
        except requests.RequestException as exc:
            st.error(f"Could not complete the prediction request: {exc}")
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(f"The API returned an invalid prediction: {exc}")

    payload = st.session_state.get("prediction")
    if payload and payload.get("trip_id") == selected_trip:
        metrics = payload.get("metrics") or {}
        st.markdown(f"### Trip `{payload['trip_id']}`")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("GPS observations", metrics.get("n_points", len(payload.get("gps", []))))
        m2.metric("Route continuity", _pct(metrics.get("route_continuity")))
        m3.metric("Classifier top-1", _pct(metrics.get("top1_acc_classifier")))
        m4.metric("HMM top-1", _pct(metrics.get("top1_acc_hmm")))
        m5.metric("Prediction time", _seconds(metrics.get("latency_s")))
        if metrics.get("top1_acc_hmm") is not None:
            st.caption(f"HMM top-1 accuracy: {_pct(metrics.get('top1_acc_hmm'))}")
        st.markdown("#### Classifier and HMM/Viterbi route comparison")
        try:
            map_obj = build_route_map(payload)
            components.html(map_obj.get_root().render(), height=590, scrolling=False)
        except (ValueError, TypeError, IndexError) as exc:
            st.warning(f"Map could not be drawn: {exc}")
        st.caption("The interface displays only prediction output. Reference routes and labels are not requested or shown here.")
else:
    st.info("Start the API, or turn on demo mode in the sidebar to preview the interface.")
