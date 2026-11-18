"""Smoke tests on dummy data.   Run:  python -m pytest -q tests/test_cv_harness.py"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("out")
    os.environ.update({"PATHSNAP_DATA": "dummy", "PATHSNAP_RESULTS_DIR": str(tmp / "results"),
                       "PATHSNAP_ARTIFACTS_DIR": str(tmp / "artifacts")})
    os.environ.pop("PATHSNAP_RESULTS_FILE", None)
    subprocess.check_call([sys.executable, str(ROOT / "tests" / "make_dummy_candidates.py")])
    return tmp


def test_metrics_basic(env):
    from src.models import metrics as M
    from src.models.common import load_partition
    c, ref = load_partition("student")
    p_perfect = c["label"].to_numpy().astype(float)
    assert M.top1_accuracy(c, p_perfect, ref) == pytest.approx(1 - M.candidate_miss_rate(c, ref), abs=1e-9)
    assert M.macro_f1(c["label"], p_perfect) == 1.0
    assert 0 < M.candidate_miss_rate(c, ref) < 0.2


def test_run_model_writes_contract_row(env):
    from sklearn.linear_model import LogisticRegression
    from src.models.cv import COLUMNS, run_model
    row = run_model("LogisticRegression", "linear", LogisticRegression(max_iter=500), verbose=False)
    df = pd.read_csv(env / "results" / "cv_results.csv", na_values=["N/A"])
    assert list(df.columns[:14]) == COLUMNS[:14]
    assert len(df) == 1 and row["st_top1"] > 0.5
    run_model("LogisticRegression", "linear", LogisticRegression(max_iter=500), verbose=False)  # re-run replaces
    assert len(pd.read_csv(env / "results" / "cv_results.csv")) == 1
    assert (env / "artifacts" / "models" / "LogisticRegression.joblib").exists()


def test_model_interface(env):
    import joblib
    from src.models.common import get_feature_cols, load_partition
    pipe = joblib.load(env / "artifacts" / "models" / "LogisticRegression.joblib")
    c, _ = load_partition("student")
    p = pipe.predict_proba(c[get_feature_cols()])[:, 1]
    assert p.shape == (len(c),) and ((p >= 0) & (p <= 1)).all()
