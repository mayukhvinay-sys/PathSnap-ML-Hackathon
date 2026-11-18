"""Group-CV harness (contract A, owner C).

    run_model(name, family, estimator, config_dict)  -> dict
fits the estimator inside Pipeline([StandardScaler, estimator]), runs GroupKFold(5) by trip on the
80 training trips, refits on all training trips, scores the 10 student_test trips, and writes one
row to results/cv_results.csv (+ results/overfit_check.csv, results/confusion/<name>.csv,
artifacts/models/<name>.joblib). Re-running a model name replaces its row.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import metrics as M
from src.models.common import (N_SPLITS, SEED, artifacts_dir, get_feature_cols, load_partition,
                               results_dir, split_xy)

COLUMNS = [
    "model", "family", "config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1", "cv_logloss",
    "st_top1", "st_macro_f1", "st_logloss", "route_continuity", "train_time_s", "infer_time_s", "seed",
    # additions (allowed by the contract: adding is fine)
    "st_precision", "st_recall", "cv_precision", "cv_recall", "train_macro_f1", "train_top1", "n_train_rows",
]


# ----------------------------------------------------------------------------- core pieces
def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", clone(estimator))])


def with_seed(estimator, seed: int = SEED):
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    return estimator


def _scores(cands, y, p, ref) -> dict:
    prec, rec = M.precision_recall(y, p)
    return {"macro_f1": M.macro_f1(y, p), "top1": M.top1_accuracy(cands, p, ref),
            "logloss": M.log_loss(y, p), "precision": prec, "recall": rec}


def _subsample_rows(idx: np.ndarray, groups: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    """Cap the number of training rows by dropping whole random trips (keeps trips intact)."""
    if not cap or len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    trips = np.unique(np.asarray(groups[idx], dtype=object).astype(str))
    rng.shuffle(trips)
    keep, n = [], 0
    for t in trips:
        c = int((groups[idx] == t).sum())
        if n + c > cap and keep:
            break
        keep.append(t)
        n += c
    return idx[np.isin(groups[idx], keep)]


def cv_scores(estimator, X, y, groups, cands, ref, *, n_splits=N_SPLITS, max_train_rows=None,
              seed=SEED, on_fold=None) -> dict:
    """GroupKFold CV. on_fold(k, running_mean_macro_f1) may raise (used by Optuna pruning)."""
    cands = cands.reset_index(drop=True)
    folds = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        tr = _subsample_rows(tr, groups, max_train_rows, seed + k)
        pipe = make_pipeline(estimator).fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        folds.append(_scores(cands.iloc[va].reset_index(drop=True), y[va], p, ref))
        if on_fold:
            on_fold(k, float(np.mean([f["macro_f1"] for f in folds])))
    df = pd.DataFrame(folds)
    return {
        "cv_macro_f1_mean": df.macro_f1.mean(), "cv_macro_f1_std": df.macro_f1.std(ddof=0),
        "cv_top1": df.top1.mean(), "cv_logloss": df.logloss.mean(),
        "cv_precision": df.precision.mean(), "cv_recall": df.recall.mean(),
        "folds": folds,
    }


def _train_eval_subset(cands: pd.DataFrame, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trips = np.asarray(cands.trip_id.unique(), dtype=object)
    rng.shuffle(trips)
    keep, n = [], 0
    counts = cands.trip_id.value_counts()
    for t in trips:
        if n >= cap:
            break
        keep.append(t)
        n += int(counts[t])
    return np.flatnonzero(cands.trip_id.isin(keep).to_numpy())


# ----------------------------------------------------------------------------- csv helpers
def _upsert(path: Path, row: dict, columns: list[str], key: str = "model") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row]).reindex(columns=columns)
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False, na_values=["N/A"])
        for c in columns:
            if c not in old:
                old[c] = np.nan
        old = old[old[key] != row[key]]
        new = pd.concat([old[columns], new], ignore_index=True)
    tmp = path.with_suffix(".tmp")
    new.to_csv(tmp, index=False, na_rep="N/A", float_format="%.6g")
    os.replace(tmp, path)


def results_path() -> Path:
    return Path(os.environ.get("PATHSNAP_RESULTS_FILE", results_dir() / "cv_results.csv"))


def merge_results() -> pd.DataFrame:
    """Merge results/cv_results*.csv (e.g. D's cv_results_D.csv) into cv_results.csv; last row per model wins."""
    rd = results_dir()
    files = sorted(rd.glob("cv_results*.csv"), key=lambda f: (f.name != "cv_results.csv", f.name))
    frames = [pd.read_csv(f, na_values=["N/A"]) for f in files]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("model", keep="last")
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    df = df[COLUMNS]
    df.to_csv(rd / "cv_results.csv", index=False, na_rep="N/A", float_format="%.6g")
    return df


# ----------------------------------------------------------------------------- public API
def run_model(name: str, family: str, estimator, config_dict: dict | None = None, *,
              n_splits: int = N_SPLITS, seed: int = SEED, max_train_rows: int | None = None,
              train_eval_rows: int = 200_000, save: bool = True, verbose: bool = True) -> dict:
    """Cross-validate + student-test one model and append its row to results/cv_results.csv."""
    feature_cols = get_feature_cols()
    train, ref_tr = load_partition("train")
    stud, ref_st = load_partition("student")
    X, y, groups = split_xy(train, feature_cols)
    Xs, ys, _ = split_xy(stud, feature_cols)
    estimator = with_seed(clone(estimator), seed)
    if config_dict is None:
        config_dict = {k: v for k, v in estimator.get_params().items()
                       if isinstance(v, (int, float, str, bool, type(None)))}
    config = dict(config_dict)
    if max_train_rows:
        config["max_train_rows_per_fold"] = max_train_rows

    if verbose:
        print(f"[{name}] {len(train):,} train rows / {train.trip_id.nunique()} trips, "
              f"{len(stud):,} student rows / {stud.trip_id.nunique()} trips")

    cv = cv_scores(estimator, X, y, groups, train, ref_tr, n_splits=n_splits,
                   max_train_rows=max_train_rows, seed=seed)

    # final fit on all training trips (timed), then student_test (timed)
    fit_idx = _subsample_rows(np.arange(len(X)), groups, max_train_rows, seed)
    pipe = make_pipeline(estimator)
    t0 = time.perf_counter()
    pipe.fit(X.iloc[fit_idx], y[fit_idx])
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    ps = pipe.predict_proba(Xs)[:, 1]
    infer_time = time.perf_counter() - t0
    st = _scores(stud, ys, ps, ref_st)
    cont = M.classifier_route_continuity(stud, ps, "student")

    # train-set scores (on a subset of whole trips so KNN etc. stay fast) for the overfitting evidence
    sub = _train_eval_subset(train, train_eval_rows, seed)
    tr_sub = train.iloc[sub].reset_index(drop=True)
    p_tr = pipe.predict_proba(X.iloc[sub])[:, 1]
    tr = _scores(tr_sub, y[sub], p_tr, ref_tr)

    row = {
        "model": name, "family": family, "config": json.dumps(config, default=str, sort_keys=True),
        "cv_macro_f1_mean": cv["cv_macro_f1_mean"], "cv_macro_f1_std": cv["cv_macro_f1_std"],
        "cv_top1": cv["cv_top1"], "cv_logloss": cv["cv_logloss"],
        "st_top1": st["top1"], "st_macro_f1": st["macro_f1"], "st_logloss": st["logloss"],
        "route_continuity": cont if cont == cont else "N/A",
        "train_time_s": train_time, "infer_time_s": infer_time, "seed": seed,
        "st_precision": st["precision"], "st_recall": st["recall"],
        "cv_precision": cv["cv_precision"], "cv_recall": cv["cv_recall"],
        "train_macro_f1": tr["macro_f1"], "train_top1": tr["top1"], "n_train_rows": len(fit_idx),
    }
    _upsert(results_path(), row, COLUMNS)

    rd = results_dir()
    _upsert(rd / "overfit_check.csv", {
        "model": name, "train_macro_f1": tr["macro_f1"], "cv_macro_f1": cv["cv_macro_f1_mean"],
        "st_macro_f1": st["macro_f1"], "train_top1": tr["top1"], "cv_top1": cv["cv_top1"],
        "st_top1": st["top1"], "gap_train_minus_cv_f1": tr["macro_f1"] - cv["cv_macro_f1_mean"],
        "gap_cv_minus_st_f1": cv["cv_macro_f1_mean"] - st["macro_f1"],
    }, ["model", "train_macro_f1", "cv_macro_f1", "st_macro_f1", "train_top1", "cv_top1", "st_top1",
        "gap_train_minus_cv_f1", "gap_cv_minus_st_f1"])
    cm = M.conf_matrix(ys, ps)
    (rd / "confusion").mkdir(exist_ok=True)
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(rd / "confusion" / f"{name}.csv")
    (rd / "cv_folds").mkdir(exist_ok=True)
    pd.DataFrame(cv["folds"]).to_csv(rd / "cv_folds" / f"{name}.csv", index_label="fold")
    if save:
        joblib.dump(pipe, artifacts_dir() / "models" / f"{name}.joblib")

    if verbose:
        print(f"[{name}] CV macro-F1 {row['cv_macro_f1_mean']:.4f}±{row['cv_macro_f1_std']:.4f} | "
              f"CV top1 {row['cv_top1']:.4f} | ST top1 {row['st_top1']:.4f} | ST macro-F1 {row['st_macro_f1']:.4f} | "
              f"ST logloss {row['st_logloss']:.4f} | train {train_time:.1f}s infer {infer_time:.2f}s")
    return row
