"""Model 3 - K-Nearest Neighbours, scaled features, distance weighted (instance-based family).
KNN scoring is slow on ~1M rows, so each fold trains on at most MAX_TRAIN_ROWS rows (whole random
trips are dropped, never single rows). The cap is recorded in the config column."""
from sklearn.neighbors import KNeighborsClassifier

from src.models.cv import run_model

MAX_TRAIN_ROWS = 300_000


def main():
    est = KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    return run_model("KNN", "instance-based", est,
                     {"n_neighbors": 25, "weights": "distance", "metric": "minkowski(p=2)",
                      "scaler": "StandardScaler"}, max_train_rows=MAX_TRAIN_ROWS)


if __name__ == "__main__":
    main()
