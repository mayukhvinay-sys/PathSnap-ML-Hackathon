# PathSnap - Member C (Models and Tuning)

Owner of: CV harness, metrics, models 1-3, model selection, Bayesian tuning, `artifacts/model.joblib`
(handoffs H3, H4, H6, H10), report sections 3 and 4.

## Layout (C's files; others' files live in the same repo)
```
configs/features.yaml            STUB for B's real file (key FEATURE_COLS)
requirements_C.txt               merge into team requirements.txt
src/models/common.py             seed, paths, data modes, loader for candidates tables
src/models/metrics.py            top1_accuracy, macro_f1, log_loss, candidate_miss_rate, precision_recall, continuity
src/models/cv.py                 run_model(name, family, estimator, config_dict) + merge_results()   [H3]
src/models/train_logreg.py       model 1
src/models/train_gnb.py          model 2
src/models/train_knn.py          model 3
src/models/train_rf.py           model 4  (D's - reference version, D overwrites)
src/models/train_hgb.py          model 5  (D's - reference version, D overwrites)
src/models/select_model.py       tables, selection, first export of artifacts/model.joblib            [H6]
src/models/plots.py              all report figures from CSVs
src/models/run_phase1.py         runs everything for Evaluation 1
src/tuning/search_space.yaml     Optuna search spaces (hgb, lgbm, rf)
src/tuning/bayes_opt.py          Optuna TPE + refit + tuned_vs_untuned + final export                  [H10]
tests/make_dummy_candidates.py   dummy candidates with the contract schema
tests/test_cv_harness.py         smoke tests
report/sections_3_4_skeleton.md  headings and where each number comes from
```
Outputs: `results/cv_results.csv, student_test_results.csv, overfit_check.csv, model_comparison.md,
selection.json, optuna_trials.csv, best_params.json, tuned_vs_untuned.csv, figures/*.png, confusion/, cv_folds/`
and `artifacts/model.joblib, scaler.joblib, model_meta.json, models/<name>.joblib`.

## Run (from the repo root; Windows: use `set VAR=value` instead of `VAR=value`)
```
pip install -r requirements_C.txt

# 0. before B's files exist (dummy data, NOT for the report)
python tests/make_dummy_candidates.py
PATHSNAP_DATA=dummy python -m pytest -q tests
PATHSNAP_DATA=dummy python -m src.models.run_phase1

# 1. when B pushes the 3-trip sample (H1):   PATHSNAP_DATA=sample python -m src.models.run_phase1 --only C
# 2. when B pushes full tables (H2): put candidates_train.parquet, candidates_student.parquet,
#    ref_arc_*.parquet in data/processed/ then:
python -m src.models.run_phase1            # models 1-5 + selection + plots + untuned model.joblib   (11:10-11:20)

# Phase 2
python -m src.tuning.bayes_opt --n-trials 30            # tunes the model named in results/selection.json (or --model hgb|lgbm|rf)
```
## Contracts I rely on
- Candidates table columns: trip_id, obs_idx, arc_id, after_gap, label + FEATURE_COLS; ref table (trip_id, obs_idx, ref_arc).
- Model interface: Pipeline([StandardScaler, estimator]) with predict_proba(X[FEATURE_COLS])[:, 1].
- Route continuity: uses D's `src.decoder.continuity.route_continuity(pred_df[obs_idx, arc_id], arcs)` once it exists,
  else a fallback in metrics.py (share of consecutive points, not across a gap, whose arcs are equal or head-to-tail connected).
  Re-run `run_phase1` after D's file lands so the column is consistent. N/A if raw `.arcs` files cannot be found
  (set PATHSNAP_RAW if the raw folder is not `Student folder/Student folder`).

## Working with D (results file)
D can call `run_model(...)` on their own laptop. To avoid git conflicts on one CSV, D may set
`PATHSNAP_RESULTS_FILE=results/cv_results_D.csv`; `select_model.py` merges every `results/cv_results*.csv`.

## Decisions you must be able to explain
- GroupKFold by trip (no trip split across train/validation) - GPS points of a trip are highly correlated.
- Primary metric = CV macro-F1 (imbalanced 0/1 labels, accuracy would be ~90% by always saying 0); top-1 per point is the route-level metric.
- Decision threshold 0.5 (common.THRESHOLD) for precision/recall/macro-F1. Top-1 and log loss do not depend on it.
- Scaler lives inside the Pipeline and is fitted on training folds only (no leakage).
- KNN is capped at 300k rows per fold (whole trips dropped) for speed; stated in its config.
- Tuned CV score is the best of N trials on the same folds -> slightly optimistic; student-test is the fair comparison.
- Student-test is used for development comparison only; evaluator-test is never touched.
