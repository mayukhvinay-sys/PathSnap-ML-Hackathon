"""Model 5 - HistGradientBoosting (boosting family).  OWNER: Member D (H5).
Reference version written by C so everything runs end to end; D may replace it."""
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.cv import run_model

PARAMS = dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, early_stopping=False)


def main():
    return run_model("HistGradientBoosting", "boosting", HistGradientBoostingClassifier(**PARAMS), dict(PARAMS))


if __name__ == "__main__":
    main()
