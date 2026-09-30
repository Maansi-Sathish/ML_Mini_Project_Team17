"""Shared evaluation utilities (written once, agreed by both people).

Rules: train on log1p(count), predict with expm1, same CV splits for everyone.
Run everything from the project root, e.g.  python -m src.models_linear
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PRED_DIR = ROOT / "predictions"
N_SPLITS = 5
SEED = 42


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["datetime"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["datetime"])
    return train, test


def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error on raw counts."""
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(np.asarray(y_true, float))) ** 2)))


def get_cv():
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def cv_rmsle(model, X, y_count, log_target=True):
    """Mean CV RMSLE. log_target=True -> fit on log1p(count), predict expm1."""
    y_count = np.asarray(y_count, float)
    scores = []
    for tr, va in get_cv().split(X):
        m = clone(model)
        y_tr = np.log1p(y_count[tr]) if log_target else y_count[tr]
        m.fit(X.iloc[tr], y_tr)
        p = m.predict(X.iloc[va])
        p = np.expm1(p) if log_target else p
        scores.append(rmsle(y_count[va], p))
    return float(np.mean(scores))


def save_result(name, score):
    """Writes results/<name>.csv with columns: model, cv_rmsle."""
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"model": [name], "cv_rmsle": [score]}).to_csv(RESULTS_DIR / f"{name}.csv", index=False)


def fit_predict_test(model, X, y_count, X_test, log_target=True):
    """Fit on all training data, return predicted counts for X_test."""
    y_count = np.asarray(y_count, float)
    m = clone(model)
    m.fit(X, np.log1p(y_count) if log_target else y_count)
    p = m.predict(X_test)
    p = np.expm1(p) if log_target else p
    return np.clip(p, 0, None)


def save_predictions(name, datetimes, counts):
    """Writes predictions/<name>.csv with columns: datetime,count."""
    PRED_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"datetime": datetimes, "count": counts}).to_csv(PRED_DIR / f"{name}.csv", index=False)
