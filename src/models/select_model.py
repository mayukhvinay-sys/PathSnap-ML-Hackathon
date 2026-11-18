"""Merge results, write comparison tables, choose the model to tune, export the first (untuned) model.

Outputs: results/cv_results.csv (merged), results/student_test_results.csv, results/model_comparison.md
(report table 3.1 layout, real numbers only), results/selection.json, results/figures/*.png,
artifacts/model.joblib + artifacts/model_meta.json (best UNTUNED model, handoff H6).

Primary metric: CV macro-F1 (mean over 5 trip-grouped folds on the 80 training trips). It is computed
only on held-out trips, is insensitive to the 1-positive-per-~10-candidates imbalance (unlike accuracy)
and is what Optuna later optimises. Student-test top-1 / log loss / continuity are reported next to it
and used as tie-breakers (a model within one CV std of the best is judged on top-1 then log loss).
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.models import plots
from src.models.common import SEED, artifacts_dir, get_feature_cols, results_dir
from src.models.cv import merge_results

TUNABLE = {"HistGradientBoosting": "hgb", "LightGBM": "lgbm", "RandomForest": "rf"}


def export_model(name: str, tuned: bool = False, extra: dict | None = None):
    pipe = joblib.load(artifacts_dir() / "models" / f"{name}.joblib")
    a = artifacts_dir()
    joblib.dump(pipe, a / "model.joblib")
    joblib.dump(pipe.named_steps["scaler"], a / "scaler.joblib")
    meta = {"model_name": name, "feature_cols": get_feature_cols(), "seed": SEED, "tuned": tuned}
    meta.update(extra or {})
    (a / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"exported {name} -> {a/'model.joblib'}")


def choose(df: pd.DataFrame) -> dict:
    d = df[~df.model.str.endswith("_tuned")].dropna(subset=["cv_macro_f1_mean"]).copy()
    best = d.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
    near = d[d.cv_macro_f1_mean >= best.cv_macro_f1_mean - best.cv_macro_f1_std]
    near = near.sort_values(["st_top1", "st_logloss"], ascending=[False, True])
    chosen = near.iloc[0]
    # candidate to tune: best tunable (boosting/ensemble) model by CV macro-F1
    tun = d[d.model.isin(TUNABLE)].sort_values("cv_macro_f1_mean", ascending=False)
    return {"primary_metric": "cv_macro_f1_mean", "best_by_primary": best.model, "chosen_untuned": chosen.model,
            "tune_model": TUNABLE[tun.iloc[0].model] if len(tun) else None,
            "n_models": int(len(d)), "within_one_std": list(near.model)}


def comparison_markdown(df: pd.DataFrame) -> str:
    cols = [("model", "Model"), ("family", "Algorithm family"), ("config", "Key configuration"),
            ("cv_macro_f1_mean", "CV macro-F1"), ("st_top1", "Top-1 acc (student-test)"),
            ("route_continuity", "Route continuity"), ("st_macro_f1", "Student-test macro-F1"),
            ("st_logloss", "Log loss"), ("infer_time_s", "Inference time (s)")]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c, _ in cols:
            v = r[c]
            if c == "cv_macro_f1_mean":
                v = f"{v:.4f} ± {r['cv_macro_f1_std']:.4f}"
            elif c == "config":
                v = str(v).replace("|", "/")[:90]
            elif isinstance(v, float):
                v = "N/A" if np.isnan(v) else f"{v:.4f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(export: bool = True):
    df = merge_results()
    rd = results_dir()
    if len(df[~df.model.str.endswith("_tuned")]) < 5:
        print(f"WARNING: only {len(df)} models in cv_results.csv; at least 5 are required before selecting.")
    df[["model", "family", "st_top1", "st_macro_f1", "st_precision", "st_recall", "st_logloss",
        "route_continuity", "infer_time_s"]].to_csv(rd / "student_test_results.csv", index=False, na_rep="N/A", float_format="%.6g")
    (rd / "model_comparison.md").write_text(comparison_markdown(df) + "\n")
    sel = choose(df)
    (rd / "selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))
    plots.make_all(df)
    if export:
        export_model(sel["chosen_untuned"])
    return sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-export", action="store_true")
    main(export=not ap.parse_args().no_export)
