"""Person A: casual vs registered experiment.

Compares two ways of predicting the total count, on the SAME 5 folds and the SAME model:
    (1) one model trained on total count
    (2) two models, one on casual and one on registered users, predictions added together
(The paper found (1) was better.)

Run from project root:
    python -m src.casual_registered            # full run
    python -m src.casual_registered --fast     # quick test on a 2500-row sample

Output: results/extra/casual_registered.csv  (NOT in results/, so the model comparison is not affected)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from src.evaluate import load_data
from src.features import build_features

OUT = Path(__file__).resolve().parents[1] / "results" / "extra"


def rmsle(actual, pred):
    pred = np.clip(pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="quick test on a small sample")
    args = ap.parse_args()

    train, _ = load_data()
    if args.fast:
        train = train.sample(2500, random_state=0).reset_index(drop=True)
    X = build_features(train, onehot_buckets=False)          # compact columns for a tree model
    total, casual, registered = (train[c].values for c in ["count", "casual", "registered"])

    scores = {"total count (one model)": [], "casual + registered (two models)": [],
              "casual only": [], "registered only": []}
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
        def predict(target):
            m = HistGradientBoostingRegressor(max_iter=150, random_state=0)
            m.fit(X.iloc[tr], np.log1p(target[tr]))
            return np.expm1(m.predict(X.iloc[va]))

        p_total, p_cas, p_reg = predict(total), predict(casual), predict(registered)
        scores["total count (one model)"].append(rmsle(total[va], p_total))
        scores["casual + registered (two models)"].append(rmsle(total[va], p_cas + p_reg))
        scores["casual only"].append(rmsle(casual[va], p_cas))
        scores["registered only"].append(rmsle(registered[va], p_reg))

    out = pd.DataFrame([{"approach": k, "cv_rmsle": round(np.mean(v), 5), "cv_std": round(np.std(v), 5)}
                        for k, v in scores.items()])
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "casual_registered.csv", index=False)
    print(out.to_string(index=False))
    print("\n(casual only / registered only are RMSLE on their own targets, for reference)")
    print(f"Saved to {OUT / 'casual_registered.csv'}")


if __name__ == "__main__":
    main()