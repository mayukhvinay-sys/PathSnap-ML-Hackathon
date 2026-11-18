"""Figures for report sections 3.3 and 4. Every figure is drawn from CSVs in results/ (real numbers only)."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.common import results_dir


def _read(name):
    p = results_dir() / name
    return pd.read_csv(p, na_values=["N/A"]) if p.exists() else None


def _save(fig, name):
    d = results_dir() / "figures"
    d.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(d / name, dpi=200)
    plt.close(fig)
    print("  figure:", d / name)


def _bar(df, col, title, ylabel, fname, err=None, best="max"):
    d = df.dropna(subset=[col])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(d.model, d[col], yerr=None if err is None else d[err], capsize=4, color="#4c78a8")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    _save(fig, fname)


def make_all(df: pd.DataFrame | None = None):
    df = df if df is not None else _read("cv_results.csv")
    if df is None or df.empty:
        print("no results to plot"); return
    base = df[~df.model.str.endswith("_tuned")]
    _bar(base, "cv_macro_f1_mean", "5-fold group CV macro-F1 (80 training trips)", "macro-F1", "cv_macro_f1.png", err="cv_macro_f1_std")
    _bar(base, "cv_top1", "5-fold group CV observation-level top-1", "top-1 accuracy", "cv_top1.png")
    _bar(base, "st_logloss", "Student-test log loss (lower is better)", "log loss", "st_logloss.png")
    _bar(base, "infer_time_s", "Inference time on the 10 student-test trips", "seconds", "infer_time.png")
    _bar(base, "train_time_s", "Final training time (80 training trips)", "seconds", "train_time.png")

    d = base.dropna(subset=["st_top1", "st_macro_f1"])
    if len(d):
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        x = np.arange(len(d)); w = 0.38
        b1 = ax.bar(x - w / 2, d.st_top1, w, label="top-1 accuracy", color="#4c78a8")
        b2 = ax.bar(x + w / 2, d.st_macro_f1, w, label="macro-F1", color="#f58518")
        ax.bar_label(b1, fmt="%.3f", fontsize=7, padding=2); ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=2)
        ax.set_xticks(x); ax.set_xticklabels(d.model, rotation=25)
        ax.set_title("Student-test performance (10 trips)"); ax.legend()
        _save(fig, "student_top1_macro_f1.png")

    # train vs CV vs student (over/underfitting evidence)
    o = _read("overfit_check.csv")
    if o is not None and not o.empty:
        o = o[~o.model.str.endswith("_tuned")]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        x = np.arange(len(o)); w = 0.27
        ax.bar(x - w, o.train_macro_f1, w, label="train", color="#54a24b")
        ax.bar(x, o.cv_macro_f1, w, label="CV", color="#4c78a8")
        ax.bar(x + w, o.st_macro_f1, w, label="student-test", color="#f58518")
        ax.set_xticks(x); ax.set_xticklabels(o.model, rotation=25); ax.set_ylabel("macro-F1")
        ax.set_title("Train vs CV vs student-test macro-F1"); ax.legend()
        _save(fig, "overfit_check.png")

    # confusion matrices (student test)
    cdir = results_dir() / "confusion"
    files = [cdir / f"{m}.csv" for m in base.model if (cdir / f"{m}.csv").exists()]
    if files:
        n = len(files); cols = min(3, n); rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, f in zip(axes.ravel(), files):
            cm = pd.read_csv(f, index_col=0).to_numpy()
            ax.axis("on"); ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v:,}", ha="center", va="center", color="black", fontsize=9)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
            ax.set_title(f.stem, fontsize=10)
        _save(fig, "confusion_matrices_student.png")

    # tuning figures
    tv = _read("tuned_vs_untuned.csv")
    if tv is not None and not tv.empty:
        d = tv[tv.variant.isin(["untuned", "tuned"])]
        metrics = [("cv_macro_f1_mean", "CV macro-F1"), ("st_top1", "ST top-1"), ("st_macro_f1", "ST macro-F1")]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(metrics)); w = 0.38
        for k, (v, col) in enumerate(zip(["untuned", "tuned"], ["#9ecae9", "#4c78a8"])):
            r = d[d.variant == v]
            if r.empty: continue
            vals = [float(r.iloc[0][m]) for m, _ in metrics]
            bars = ax.bar(x + (k - .5) * w, vals, w, label=v, color=col)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in metrics]); ax.legend()
        ax.set_title("Tuned vs untuned (same folds, same partitions)")
        _save(fig, "tuned_vs_untuned.png")
    t = _read("optuna_trials.csv")
    if t is not None and not t.empty and "value" in t:
        t = t.dropna(subset=["value"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t.number, t.value, "o", alpha=.6, label="trial")
        ax.plot(t.number, t.value.cummax(), "-", color="#e45756", label="best so far")
        ax.set_xlabel("trial"); ax.set_ylabel("CV macro-F1"); ax.legend(); ax.set_title("Optuna TPE optimisation history")
        _save(fig, "optuna_history.png")


if __name__ == "__main__":
    make_all()
