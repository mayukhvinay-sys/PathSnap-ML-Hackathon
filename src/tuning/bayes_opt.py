
"""
PathSnap — Optimized Bayesian Hyperparameter Tuning
Owner: Member C

Features:
- Optuna TPE Bayesian optimization
- 5-fold GroupKFold by trip_id
- Whole-trip row subsampling for faster CV
- Persistent SQLite study; resumes after Colab disconnects
- Saves trial results after every completed/pruned trial
- Compares tuned vs untuned models
- Exports the selected model

Examples:
    python -m src.tuning.bayes_opt --model rf --n-trials 10
    python -m src.tuning.bayes_opt --model rf --n-trials 20 --timeout 1800
    python -m src.tuning.bayes_opt --model hgb --n-trials 15
    python -m src.tuning.bayes_opt --model rf --n-trials 10 --max-train-rows 300000

The student-test partition is NEVER used by the optimization objective.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import optuna
import pandas as pd
import yaml

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from src.models import plots
from src.models.common import (
    ROOT,
    SEED,
    get_feature_cols,
    load_partition,
    results_dir,
    split_xy,
)
from src.models.cv import (
    cv_scores,
    run_model,
    with_seed,
)
from src.models.select_model import export_model


# ---------------------------------------------------------
# DEFAULT SETTINGS
# ---------------------------------------------------------

DEFAULT_MAX_TRAIN_ROWS = 300_000
DEFAULT_TRIALS = 15
DEFAULT_TIMEOUT = 1800

N_SPLITS = 5
N_STARTUP_TRIALS = 5


# ---------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------

def get_estimator_class(model_key: str):
    if model_key == "rf":
        return RandomForestClassifier

    if model_key == "hgb":
        return HistGradientBoostingClassifier

    if model_key == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "LightGBM is not installed. Run: pip install lightgbm"
            ) from exc

    raise ValueError(f"Unsupported model: {model_key}")


# ---------------------------------------------------------
# PARAMETER SUGGESTION
# ---------------------------------------------------------

def suggest_parameters(trial: optuna.Trial, search_space: dict) -> dict:
    params = {}

    for name, spec in search_space.items():

        param_type = spec["type"]

        if param_type == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )

        elif param_type == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                log=bool(spec.get("log", False)),
            )

        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(
                name,
                spec["choices"],
            )

        else:
            raise ValueError(
                f"Unknown parameter type '{param_type}' for {name}"
            )

    return params


# ---------------------------------------------------------
# OUTPUT HELPERS
# ---------------------------------------------------------

def save_trial_outputs(study, results_path: Path):
    """Save trial history and current best parameters."""

    results_path.mkdir(parents=True, exist_ok=True)

    trials_path = results_path / "optuna_trials.csv"

    study.trials_dataframe().to_csv(
        trials_path,
        index=False,
    )

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if completed:
        best = study.best_trial

        best_info = {
            "best_value_cv_macro_f1": best.value,
            "best_params": best.params,
            "number_of_trials": len(study.trials),
            "completed_trials": len(completed),
            "pruned_trials": sum(
                t.state == optuna.trial.TrialState.PRUNED
                for t in study.trials
            ),
        }

        (results_path / "best_params.json").write_text(
            json.dumps(best_info, indent=4, default=str)
        )


def incremental_save_callback(results_path: Path):

    def callback(study, trial):
        save_trial_outputs(study, results_path)

        print(
            f"\nTrial {trial.number} finished: "
            f"state={trial.state.name}, "
            f"value={trial.value}"
        )

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if completed:
            print(
                f"Current best CV Macro-F1: "
                f"{study.best_value:.5f}"
            )
        else:
            print("No completed trials yet.")

    return callback

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="PathSnap optimized Bayesian hyperparameter tuning"
    )

    parser.add_argument(
        "--model",
        choices=["rf", "hgb", "lgbm"],
        default="rf",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_TRIALS,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
    )

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=DEFAULT_MAX_TRAIN_ROWS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Create a new study instead of resuming an existing one.",
    )

    args = parser.parse_args()

    if args.n_trials < 1:
        parser.error("--n-trials must be at least 1")

    if args.max_train_rows < 1:
        parser.error("--max-train-rows must be at least 1")

    # -----------------------------------------------------
    # PATHS AND CONFIGURATION
    # -----------------------------------------------------

    rd = results_dir()
    rd.mkdir(parents=True, exist_ok=True)

    tuning_dir = rd / "optuna"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    config_path = ROOT / "src" / "tuning" / "search_space.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Search-space configuration not found: {config_path}"
        )

    with config_path.open("r") as f:
        all_spaces = yaml.safe_load(f)

    if args.model not in all_spaces:
        raise ValueError(
            f"Model '{args.model}' is missing from {config_path}"
        )

    cfg = all_spaces[args.model]

    model_name = cfg["name"]
    family = cfg["family"]

    param_space = cfg["params"]
    fixed_params = cfg.get("fixed", {})

    estimator_cls = get_estimator_class(args.model)

    # -----------------------------------------------------
    # DATA LOADING
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("PATHSNAP — BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 65)

    print(f"Model:             {model_name}")
    print(f"Trials requested:  {args.n_trials}")
    print(f"Timeout:           {args.timeout} seconds")
    print(f"Training row cap:  {args.max_train_rows:,}")
    print(f"CV folds:          {N_SPLITS}")
    print(f"Seed:              {args.seed}")

    train, ref_train = load_partition("train")

    X, y, groups = split_xy(
        train,
        get_feature_cols(),
    )

    print("\nTraining data:")
    print(f"Rows:              {len(train):,}")
    print(f"Trips:             {train['trip_id'].nunique()}")
    print(f"Features:          {X.shape[1]}")

    if train["trip_id"].nunique() < N_SPLITS:
        raise ValueError(
            "Not enough unique trips for the requested GroupKFold."
        )

    # -----------------------------------------------------
    # BASELINE
    # -----------------------------------------------------

    cv_path = rd / "cv_results.csv"

    if cv_path.exists():
        existing = pd.read_csv(cv_path, na_values=["N/A"])
    else:
        existing = pd.DataFrame(columns=["model"])

    if "model" not in existing.columns:
        existing["model"] = pd.Series(dtype=str)

    if model_name not in set(existing["model"]):

        print(
            f"\nNo baseline found for {model_name}."
            "\nRunning untuned baseline first..."
        )

        run_model(
            model_name,
            family,
            estimator_cls(**fixed_params),
            dict(fixed_params, note="library defaults"),
            n_splits=N_SPLITS,
            seed=args.seed,
            max_train_rows=args.max_train_rows,
            save=True,
        )

    else:
        print(f"\nExisting baseline found: {model_name}")

        # Check whether the existing baseline used the same cap.
        baseline = existing[
            existing["model"] == model_name
        ].iloc[0]

        try:
            baseline_config = json.loads(baseline["config"])

            previous_cap = baseline_config.get(
                "max_train_rows_per_fold"
            )

            if (
                previous_cap is not None
                and previous_cap != args.max_train_rows
            ):
                warnings.warn(
                    "Baseline and tuning row caps differ. "
                    "The comparison may not be fair."
                )

        except (ValueError, TypeError, KeyError):
            warnings.warn(
                "Could not verify the baseline training row cap."
            )

    # -----------------------------------------------------
    # OPTUNA STORAGE
    # -----------------------------------------------------

    study_name = f"pathsnap_{args.model}"

    database_path = tuning_dir / f"{study_name}.db"

    storage_url = f"sqlite:///{database_path.as_posix()}"

    if args.no_resume and database_path.exists():
        database_path.unlink()

    print("\nOptuna database:")
    print(database_path)

    # -----------------------------------------------------
    # OBJECTIVE FUNCTION
    # -----------------------------------------------------

    def objective(trial: optuna.Trial) -> float:

        params = suggest_parameters(
            trial,
            param_space,
        )

        estimator = estimator_cls(
            **params,
            **fixed_params,
        )

        estimator = with_seed(
            estimator,
            args.seed,
        )

        trial_start = time.perf_counter()

        print(
            f"\nStarting trial {trial.number + 1}"
            f"/{args.n_trials}"
        )

        print("Parameters:", params)

        def report_fold(fold_number, running_score):

            trial.report(
                running_score,
                step=fold_number,
            )

            # Prune only after enough folds have been evaluated.
            if trial.should_prune():
                raise optuna.TrialPruned()

        # IMPORTANT:
        # - GroupKFold remains grouped by trip_id.
        # - The row cap drops whole trips, not individual candidate rows.
        # - Student-test data is not loaded or evaluated here.

        scores = cv_scores(
            estimator,
            X,
            y,
            groups,
            train,
            ref_train,
            n_splits=N_SPLITS,
            max_train_rows=args.max_train_rows,
            seed=args.seed,
            on_fold=report_fold,
        )

        duration = time.perf_counter() - trial_start

        trial.set_user_attr(
            "cv_macro_f1_std",
            float(scores["cv_macro_f1_std"]),
        )

        trial.set_user_attr(
            "cv_top1",
            float(scores["cv_top1"]),
        )

        trial.set_user_attr(
            "cv_logloss",
            float(scores["cv_logloss"]),
        )

        trial.set_user_attr(
            "cv_precision",
            float(scores["cv_precision"]),
        )

        trial.set_user_attr(
            "cv_recall",
            float(scores["cv_recall"]),
        )

        trial.set_user_attr(
            "duration_s",
            float(duration),
        )

        print(
            f"Trial {trial.number} | "
            f"Macro-F1={scores['cv_macro_f1_mean']:.5f} | "
            f"Top-1={scores['cv_top1']:.5f} | "
            f"Time={duration:.1f}s"
        )

        return float(scores["cv_macro_f1_mean"])

    # -----------------------------------------------------
    # CREATE OR RESUME STUDY
    # -----------------------------------------------------

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=N_STARTUP_TRIALS,
        multivariate=True,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=2,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=not args.no_resume,
    )

    print("\nStudy loaded.")
    print(f"Previous trials: {len(study.trials)}")

    # -----------------------------------------------------
    # RUN OPTIMIZATION
    # -----------------------------------------------------

    print("\nStarting optimization...")

    optimization_start = time.perf_counter()

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[
            incremental_save_callback(rd),
        ],
        gc_after_trial=True,
        show_progress_bar=False,
        n_jobs=1,
    )

    optimization_time = (
        time.perf_counter() - optimization_start
    )

    # Final save.
    save_trial_outputs(study, rd)

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        raise RuntimeError(
            "No successful Optuna trials. "
            "Inspect the error messages and data configuration."
        )

    # -----------------------------------------------------
    # BEST PARAMETERS
    # -----------------------------------------------------

    best_trial = study.best_trial
    best_params = best_trial.params

    best_estimator = estimator_cls(
        **best_params,
        **fixed_params,
    )

    best_estimator = with_seed(
        best_estimator,
        args.seed,
    )

    best_info = {
        "model": model_name,
        "model_key": args.model,
        "best_value_cv_macro_f1": study.best_value,
        "best_params": best_params,
        "fixed_params": fixed_params,
        "max_train_rows_per_fold": args.max_train_rows,
        "cv_folds": N_SPLITS,
        "seed": args.seed,
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": sum(
            t.state == optuna.trial.TrialState.PRUNED
            for t in study.trials
        ),
        "optimization_time_seconds": optimization_time,
    }

    (rd / "best_params.json").write_text(
        json.dumps(
            best_info,
            indent=4,
            default=str,
        )
    )

    print("\n" + "=" * 65)
    print("OPTIMIZATION COMPLETED")
    print("=" * 65)

    print(
        f"Best CV Macro-F1: {study.best_value:.5f}"
    )

    print("Best parameters:")
    print(
        json.dumps(
            best_params,
            indent=4,
        )
    )

    print(
        f"Optimization time: {optimization_time:.1f} seconds"
    )

    # -----------------------------------------------------
    # REFIT AND STUDENT-TEST EVALUATION
    # -----------------------------------------------------

    tuned_name = f"{model_name}_tuned"

    print("\nEvaluating the best tuned model...")

    tuned_result = run_model(
        tuned_name,
        family,
        best_estimator,
        dict(best_params, **fixed_params),
        n_splits=N_SPLITS,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        save=True,
    )

    # -----------------------------------------------------
    # TUNED VS UN-TUNED COMPARISON
    # -----------------------------------------------------

    all_results = pd.read_csv(
        cv_path,
        na_values=["N/A"],
    )

    untuned_rows = all_results[
        all_results["model"] == model_name
    ]

    tuned_rows = all_results[
        all_results["model"] == tuned_name
    ]

    if untuned_rows.empty or tuned_rows.empty:
        raise RuntimeError(
            "Could not find both tuned and untuned result rows."
        )

    untuned = untuned_rows.iloc[-1]
    tuned = tuned_rows.iloc[-1]

    compare_columns = [
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "cv_top1",
        "cv_logloss",
        "st_top1",
        "st_macro_f1",
        "st_logloss",
        "infer_time_s",
        "train_time_s",
    ]

    comparison = []

    for variant, row in [
        ("untuned", untuned),
        ("tuned", tuned),
    ]:
        entry = {
            "variant": variant,
            "model": row["model"],
        }

        for col in compare_columns:
            entry[col] = row[col]

        comparison.append(entry)

    delta = {
        "variant": "delta (tuned - untuned)",
        "model": model_name,
    }

    for col in compare_columns:
        if pd.notna(tuned[col]) and pd.notna(untuned[col]):
            delta[col] = tuned[col] - untuned[col]
        else:
            delta[col] = None

    comparison.append(delta)

    comparison_df = pd.DataFrame(comparison)

    comparison_df.to_csv(
        rd / "tuned_vs_untuned.csv",
        index=False,
        na_rep="N/A",
        float_format="%.6g",
    )

    print("\nTUNED VS UNTUNED:")
    print(comparison_df.to_string(index=False))

    # -----------------------------------------------------
    # EXPORT SELECTED MODEL
    # -----------------------------------------------------

    # The primary selection rule is CV Macro-F1.
    # Student-test metrics are reported independently.
    use_tuned = (
        float(tuned["cv_macro_f1_mean"])
        >= float(untuned["cv_macro_f1_mean"])
    )

    selected_name = (
        tuned_name if use_tuned else model_name
    )

    print("\nSelected model:", selected_name)

    export_model(
        selected_name,
        tuned=use_tuned,
        extra={
            "params": best_params if use_tuned else "untuned baseline",
            "tuning_trials": len(study.trials),
            "max_train_rows": args.max_train_rows,
            "selection_metric": "CV Macro-F1",
        },
    )

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------

    plots.make_all(all_results)

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("MEMBER C — DELIVERABLES")
    print("=" * 65)

    expected_outputs = [
        rd / "optuna_trials.csv",
        rd / "best_params.json",
        rd / "tuned_vs_untuned.csv",
        rd / "overfit_check.csv",
        tuning_dir / f"{study_name}.db",
    ]

    for path in expected_outputs:
        print(
            f"{'[OK]' if path.exists() else '[MISSING]'} "
            f"{path}"
        )

    print("\nSelected model:", selected_name)
    print("Best CV Macro-F1:", round(study.best_value, 5))
    print("Completed trials:", len(completed_trials))
    print("Tuning time:", round(optimization_time, 2), "seconds")

    print("\nMember C optimization finished.")


if __name__ == "__main__":
    main()
