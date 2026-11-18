"""API settings. Override with environment variables when running on Laptop 1.

PATHSNAP_TRIP_DIR   folder that contains one sub-folder per trip (<id>/<id>.track ...).
                    Default: the student_test partition (development).
PATHSNAP_MODEL      saved scikit-learn model (joblib). Default: artifacts/models/RandomForest.joblib
"""
import os
from pathlib import Path

from src.candidates._io import REPO_ROOT, data_root, load_config

TRIP_DIR = Path(os.environ.get("PATHSNAP_TRIP_DIR", str(data_root() / "student_test")))
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL",
                                 str(REPO_ROOT / "artifacts" / "models" / "RandomForest.joblib")))
HMM_PARAMS_PATH = REPO_ROOT / "artifacts" / "hmm_params.json"

# Candidate settings must match the ones used to build the training tables.
CANDIDATE_CFG = load_config("candidates.yaml")
