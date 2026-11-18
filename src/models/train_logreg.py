"""Model 1 - Logistic Regression (linear / probabilistic family)."""
from sklearn.linear_model import LogisticRegression

from src.models.cv import run_model


def main():
    est = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)  # L2 by default
    return run_model("LogisticRegression", "linear/probabilistic", est,
                     {"C": 1.0, "regularisation": "l2 (default)", "solver": "lbfgs", "max_iter": 1000, "scaler": "StandardScaler"})


if __name__ == "__main__":
    main()
