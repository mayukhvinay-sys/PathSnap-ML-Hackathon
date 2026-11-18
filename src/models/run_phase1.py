"""Run models 1-3 (C) and, if present, 4-5 (D's reference scripts), then select + plot.
    python -m src.models.run_phase1            # all five
    python -m src.models.run_phase1 --only C   # just LogReg, GNB, KNN
"""
import argparse

from src.models import select_model, train_gnb, train_hgb, train_knn, train_logreg, train_rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["C", "all"], default="all")
    a = ap.parse_args()
    for mod in (train_logreg, train_gnb, train_knn):
        mod.main()
    if a.only == "all":
        for mod in (train_rf, train_hgb):
            mod.main()
    select_model.main()


if __name__ == "__main__":
    main()
