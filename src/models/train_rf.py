"""Model 4 - Random Forest (ensemble family).  OWNER: Member D (H5).
Reference version written by C so everything runs end to end; D may replace it."""
from sklearn.ensemble import RandomForestClassifier

from src.models.cv import run_model


def main():
    est = RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=5, n_jobs=-1)
    return run_model("RandomForest", "ensemble", est,
                     {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5, "criterion": "gini"})


if __name__ == "__main__":
    main()
