"""Model 2 - Gaussian Naive Bayes (probabilistic family)."""
from sklearn.naive_bayes import GaussianNB

from src.models.cv import run_model


def main():
    est = GaussianNB(var_smoothing=1e-9)
    return run_model("GaussianNB", "linear/probabilistic", est,
                     {"var_smoothing": 1e-9, "priors": "empirical", "scaler": "StandardScaler"})


if __name__ == "__main__":
    main()
