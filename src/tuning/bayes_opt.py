"""Bayesian hyper-parameter optimisation (Optuna TPE) with group-aware CV. Handoffs H6 (tuned model) + H10.

    python -m src.tuning.bayes_opt                      # model from results/selection.json (default hgb), 30 trials
    python -m src.tuning.bayes_opt --model lgbm --n-trials 40 --timeout 900

Steps: (1) make sure an untuned baseline row exists, (2) run TPE on 5-fold GroupKFold CV macro-F1 of the
80 training trips, (3) save every trial, (4) refit best params and evaluate on the 10 student-test trips with the
same harness as the untuned model, (5) write tuned_vs_untuned.csv, (6) export artifacts/model.joblib
(tuned if its CV macro-F1 is not worse than untuned, otherwise the untuned model is kept).
Note: the tuned model's CV score is the best of N trials on the same folds, so it is slightly optimistic;
the student-test numbers are the unbiased comparison.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.models import plots
from src.models.common import ROOT, SEED, get_feature_cols, load_partition, results_dir, split_xy
from src.models.cv import COLUMNS, cv_scores, run_model, with_seed
from src.models.select_model import export_model


def _estimator_cls(key: str):
    if key == "hgb":
        return HistGradientBoostingClassifier
    if key == "rf":
        return RandomForestClassifier
    if key == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    raise ValueError(key)


def suggest(trial: optuna.Trial, space: dict) -> dict:
    out = {}
    for k, s in space.items():
        if s["type"] == "float":
            out[k] = trial.suggest_float(k, float(s["low"]), float(s["high"]), log=bool(s.get("log", False)))
        elif s["type"] == "int":
            out[k] = trial.suggest_int(k, int(s["low"]), int(s["high"]), log=bool(s.get("log", False)))
        else:
            out[k] = trial.suggest_categorical(k, s["choices"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=["hgb", "lgbm", "rf"])
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=None, help="seconds for the whole study")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    rd = results_dir()
    sel_path = rd / "selection.json"
    key = a.model or (json.loads(sel_path.read_text()).get("tune_model") if sel_path.exists() else None) or "hgb"
    cfg = yaml.safe_load((ROOT / "src" / "tuning" / "search_space.yaml").read_text())[key]
    cls, space, fixed = _estimator_cls(key), cfg["params"], cfg.get("fixed", {})
    name, family = cfg["name"], cfg["family"]
    print(f"Tuning {name} | objective: CV macro-F1 (GroupKFold by trip) | {a.n_trials} trials | seed {a.seed}")

    # 1) untuned baseline with default-ish params, same harness
    res = pd.read_csv(rd / "cv_results.csv", na_values=["N/A"]) if (rd / "cv_results.csv").exists() else pd.DataFrame(columns=["model"])
    if name not in set(res.model):
        print(f"no untuned row for {name}; running it with library defaults first")
        run_model(name, family, cls(**fixed), dict(fixed, note="library defaults"))

    # 2) Optuna study
    train, ref_tr = load_partition("train")
    X, y, groups = split_xy(train, get_feature_cols())

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space)
        est = with_seed(cls(**params, **fixed), a.seed)

        def on_fold(k, running):
            trial.report(running, k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        t0 = time.perf_counter()
        s = cv_scores(est, X, y, groups, train, ref_tr, on_fold=on_fold)
        trial.set_user_attr("cv_top1", s["cv_top1"])
        trial.set_user_attr("cv_logloss", s["cv_logloss"])
        trial.set_user_attr("cv_macro_f1_std", s["cv_macro_f1_std"])
        trial.set_user_attr("duration_s", time.perf_counter() - t0)
        return s["cv_macro_f1_mean"]

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="maximize", study_name=f"pathsnap_{key}",
                                sampler=optuna.samplers.TPESampler(seed=a.seed, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=a.n_trials, timeout=a.timeout, gc_after_trial=True)
    tuning_cost = time.perf_counter() - t0

    # 3) keep the study
    trials = study.trials_dataframe()
    trials.to_csv(rd / "optuna_trials.csv", index=False)
    joblib.dump(study, rd / "optuna_study.pkl")
    best = study.best_params
    (rd / "best_params.json").write_text(json.dumps({"model": name, "best_value_cv_macro_f1": study.best_value,
                                                      "best_params": best, "fixed": fixed, "n_trials": len(study.trials),
                                                      "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
                                                      "seed": a.seed, "tuning_cost_s": tuning_cost}, indent=2, default=str))
    print("best CV macro-F1:", round(study.best_value, 5), "\nbest params:", best)

    # 4) refit + evaluate with the same harness as the untuned model
    tuned_name = f"{name}_tuned"
    run_model(tuned_name, family, cls(**best, **fixed), dict(best, **fixed))

    # 5) tuned vs untuned
    all_res = pd.read_csv(rd / "cv_results.csv", na_values=["N/A"])
    u = all_res[all_res.model == name].iloc[0]
    t = all_res[all_res.model == tuned_name].iloc[0]
    cols = ["cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1", "st_top1", "st_macro_f1", "st_logloss",
            "route_continuity", "infer_time_s", "train_time_s"]
    rows = [dict(variant="untuned", model=name, **{c: u[c] for c in cols}, n_trials="N/A", tuning_cost_s="N/A"),
            dict(variant="tuned", model=tuned_name, **{c: t[c] for c in cols}, n_trials=len(study.trials),
                 tuning_cost_s=round(tuning_cost, 1))]
    rows.append(dict(variant="delta (tuned-untuned)", model=name,
                     **{c: (t[c] - u[c]) if pd.notna(t[c]) and pd.notna(u[c]) else "N/A" for c in cols},
                     n_trials="", tuning_cost_s=""))
    pd.DataFrame(rows).to_csv(rd / "tuned_vs_untuned.csv", index=False, na_rep="N/A", float_format="%.6g")
    print(pd.DataFrame(rows).T)

    # 6) export (selection rule uses CV only; student-test is reported, not used to pick)
    use_tuned = bool(t.cv_macro_f1_mean >= u.cv_macro_f1_mean)
    export_model(tuned_name if use_tuned else name, tuned=use_tuned,
                 extra={"params": best if use_tuned else "library defaults", "tuning_trials": len(study.trials)})
    plots.make_all(all_res)


if __name__ == "__main__":
    main()
