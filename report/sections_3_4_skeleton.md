# Report sections 3 and 4 (Member C) - SKELETON
Write the analysis yourself from the CSVs. Every number must exist in results/*.csv (rule: real numbers only).
Drafting tools exist, but the integrity policy says you must be able to explain every sentence in the viva.

## 3.1 Overall model comparison
Paste results/model_comparison.md (auto-generated). Add one sentence: which partition each column uses
(CV = 80 training trips, 5-fold GroupKFold by trip; Student-test = 10 student_test trips).
State the PRIMARY METRIC and why: results/selection.json, docstring of src/models/select_model.py.
Explain N/A cells (e.g. route continuity if the arcs file was unavailable).

## 3.2 One subsection per model (repeat 5 times: LogisticRegression, GaussianNB, KNN, RandomForest, HistGradientBoosting)
### Model N - <name> (<family>)
A. Model understanding
 1. Why suitable for PathSnap (candidate arc vs not, features are tabular, need calibrated probabilities for the HMM emission)?
 2. Strengths and limitations (link to what you saw: log loss, training/inference time, imbalance 1 positive in ~10 candidates).
B. Training and configuration
 5. Important hyperparameters + initial values -> `config` column in results/cv_results.csv
 6. Loss / objective -> (LogReg: log loss + L2; GNB: max. likelihood / no iterative loss; KNN: none, lazy learner; RF: gini; HGB: log loss)
C. Results
 7. Student-test performance -> results/student_test_results.csv
 8. Top-1, macro-F1, log loss on student-test -> same file
 9. Over/underfitting evidence -> results/overfit_check.csv (train vs CV vs student-test), results/cv_folds/<model>.csv (fold stability)
No loss curves for models without epoch histories (do not draw artificial ones).

## 3.3 Figures (results/figures/)
cv_macro_f1.png, cv_top1.png, student_top1_macro_f1.png, st_logloss.png, infer_time.png, train_time.png,
overfit_check.png, confusion_matrices_student.png.  Add a selection paragraph that refers to them.

## 4 Bayesian hyper-parameter optimisation (chosen model)
1. Hyperparameters searched          -> src/tuning/search_space.yaml
2. Ranges / distributions (log scale where used, why)
3. Objective                          -> mean 5-fold GroupKFold macro-F1 on the 80 training trips
4. Number of trials (and pruned)      -> results/best_params.json, results/optuna_trials.csv
5. Best values                        -> results/best_params.json
6. Tuned vs untuned                   -> results/tuned_vs_untuned.csv, figures/tuned_vs_untuned.png, figures/optuna_history.png
   Mention: tuned CV score is the best of N trials on the same folds (optimistic) - the student-test numbers are the fair comparison.
   Mention computational cost: tuning_cost_s, train_time_s, infer_time_s.
