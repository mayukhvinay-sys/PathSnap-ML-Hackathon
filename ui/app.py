from __future__ import annotations

import json
import os
from pathlib import Path

import folium
import requests
import streamlit as st
from folium import plugins
from streamlit_folium import st_folium


ROOT = Path(__file__).resolve().parents[1]
MOCK_PATH = Path(__file__).with_name("mock_predict_response.json")
DEFAULT_API = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PathSnap Route Viewer", page_icon="🧭", layout="wide")
st.markdown("""
<style>
  .stApp, [data-testid="stAppViewContainer"] { background:#f4f7fb; color:#172b4d; }
  .stApp, .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2,
  .stApp h3, .stApp h4, .stApp li, .stApp small { color:#172b4d; }
  [data-testid="stHeader"] { background:rgba(244,247,251,.96); }
  [data-testid="stSidebar"] { background:#eef3f8; border-right:1px solid #dce5ef; }
  [data-testid="stSidebar"] * { color:#172b4d !important; }
  .hero { padding:1.25rem 1.6rem; border-radius:18px; color:#fff !important;
    background:linear-gradient(115deg,#102a43,#176b87 65%,#36a3a5); margin-bottom:1rem; }
  .hero-title,.hero p { color:#fff !important; -webkit-text-fill-color:#fff !important; margin:0; }
  .hero-title { font-size:2rem; line-height:1.15; font-weight:750; letter-spacing:-.025em; }
  .hero p { margin-top:.35rem; opacity:.92; }
  [data-testid="stMetric"] { background:#fff; border:1px solid #dce5ef;
    padding:.85rem 1rem; border-radius:14px; box-shadow:0 3px 10px #172b4d0a; }
  [data-testid="stMetricLabel"] p { color:#52657a !important; font-size:.86rem; }
  [data-testid="stMetricValue"] { color:#102a43 !important; font-weight:700; }
  [data-baseweb="select"] > div { background:#fff; border-color:#cbd7e4; }
  [data-baseweb="select"] * { color:#172b4d !important; }
  [data-testid="stTextInput"] input { color:#172b4d !important; background:#fff !important; }
  [data-testid="stButton"] button[kind="primary"] { background:#126782; border-color:#126782; color:#fff !important; }
  [data-testid="stButton"] button[kind="primary"] p { color:#fff !important; font-weight:600; }
  [data-testid="stAlert"] { border-radius:12px; }
  iframe { border:1px solid #dce5ef !important; border-radius:16px; overflow:hidden; }
</style>
<div class="hero"><div role="heading" aria-level="1" class="hero-title" style="color:#fff!important;-webkit-text-fill-color:#fff!important">PathSnap</div><p style="color:#f1f7fb!important;-webkit-text-fill-color:#f1f7fb!important">GPS map matching · Route evaluation</p></div>
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


def _routes_match(path_a, path_b) -> bool:
    """Detect identical route geometry while ignoring tiny serialization noise."""
    def normalized(points):
        return [(round(float(p[0]), 6), round(float(p[1]), 6))
                for p in points if isinstance(p, (list, tuple)) and len(p) == 2]

    a, b = normalized(path_a or []), normalized(path_b or [])
    return bool(a) and (a == b or a == list(reversed(b)))
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
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True,
                   prefer_canvas=True, zoom_control=True)
    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    if len(gps) > 1000:
        plugins.FastMarkerCluster(gps, name="GPS observations").add_to(gps_layer)
    else:
        for point in gps:
            folium.CircleMarker(point, radius=3, color="#2563eb", fill=True,
                                fill_color="#2563eb", fill_opacity=.72, weight=1).add_to(gps_layer)
    if len(gps) > 1:
        folium.PolyLine(gps, color="#2563eb", weight=2, opacity=.48,
                        smooth_factor=2.5, line_cap="round", line_join="round",
                        tooltip="Observed GPS sequence").add_to(gps_layer)
    gps_layer.add_to(m)
    if classifier_path:
        classifier = folium.FeatureGroup(name="Classifier-only route", show=False)
        folium.PolyLine(classifier_path, color="#7c3aed", weight=4, opacity=.92,
                        smooth_factor=2.5, line_cap="round", line_join="round",
                        dash_array="8, 7",
                        tooltip="Classifier-only route").add_to(classifier)
        classifier.add_to(m)
    if path:
        predicted = folium.FeatureGroup(name="HMM/Viterbi route", show=True)
        folium.PolyLine(path, color="#d97706", weight=6, opacity=.98,
                        smooth_factor=2.5, line_cap="round", line_join="round",
                        tooltip="HMM/Viterbi predicted route").add_to(predicted)
        folium.Marker(path[0], tooltip="HMM route start", icon=folium.Icon(color="green", icon="play")).add_to(predicted)
        folium.Marker(path[-1], tooltip="HMM route end", icon=folium.Icon(color="red", icon="stop")).add_to(predicted)
        predicted.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[min(p[0] for p in coords), min(p[1] for p in coords)],
                  [max(p[0] for p in coords), max(p[1] for p in coords)]],
                 padding=(26, 26), max_zoom=16)
    plugins.Fullscreen(position="topright").add_to(m)
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
        st.session_state.pop("prediction", None)
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
        with st.expander("What am I seeing?", expanded=False):
            st.markdown(
                "- **GPS observations (blue):** the recorded locations, including measurement noise.\n"
                "- **Classifier-only route (purple dashed):** the top-scoring road candidate chosen independently for each observation. It can jump between roads.\n"
                "- **HMM/Viterbi route (orange):** a sequence of road candidates selected using both classifier scores and road-to-road continuity. It aims to form a more plausible continuous route."
            )
        if _routes_match(payload.get("pred_path"), payload.get("classifier_path")):
            st.info("For this trip, the classifier and HMM/Viterbi produced the same route geometry, so those two route lines overlap. Toggle either layer in the map to inspect it independently.")
        try:
            map_obj = build_route_map(payload)
            st_folium(map_obj, height=640, use_container_width=True,
                      returned_objects=[], key=f"route_map_{selected_trip}")
        except (ValueError, TypeError, IndexError) as exc:
            st.warning(f"Map could not be drawn: {exc}")
        st.caption("The interface displays only prediction output. Reference routes and labels are not requested or shown here.")
else:
    st.info("Start the API, or turn on demo mode in the sidebar to preview the interface.")
