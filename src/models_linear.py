"""Person A: Linear regression, Poisson GLM (elastic net), PCR, SVR.

Run from project root:
    python -m src.models_linear                 # all four, default params
    python -m src.models_linear --tune          # small CV grid search first
    python -m src.models_linear --models svr,pcr
Each model writes results/<name>.csv and predictions/<name>.csv.
"""
import argparse
import itertools
import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.evaluate import cv_rmsle, fit_predict_test, load_data, save_predictions, save_result
from src.features import build_features

# Update these after running --tune
BEST_PARAMS = {
    "pcr": {"n_components": None},                       # None = all components
    "svr": {"C": 10.0, "epsilon": 0.1, "gamma": "scale"},
    "poisson": {"alpha": 1e-3, "l1_wt": 0.5},
}


class PoissonElasticNet(BaseEstimator, RegressorMixin):
    """Poisson GLM (log link) with elastic-net penalty, glmnet-style.

    Outer loop = IRLS (Newton) for the Poisson likelihood; inner step = weighted
    sklearn ElasticNet on the working response. Fits on RAW counts, standardises
    features internally, intercept unpenalised. l1_wt=1 -> lasso, 0 -> ridge.
    """

    def __init__(self, alpha=1e-3, l1_wt=0.5, max_iter=25, tol=1e-5):
        self.alpha = alpha
        self.l1_wt = l1_wt
        self.max_iter = max_iter
        self.tol = tol

    def _scale(self, X):
        return (np.asarray(X, float) - self.mu_) / self.sd_

    def fit(self, X, y):
        Xa = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu_, self.sd_ = Xa.mean(0), Xa.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        Z = self._scale(Xa)
        eta = np.full(len(y), np.log(y.mean() + 1e-6))
        b0, coef = eta[0], np.zeros(Z.shape[1])
        for _ in range(self.max_iter):
            mu = np.exp(eta)
            z = eta + (y - mu) / mu                       # working response
            en = ElasticNet(alpha=self.alpha, l1_ratio=max(self.l1_wt, 1e-3),
                            max_iter=2000, tol=1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                en.fit(Z, z, sample_weight=mu)            # weights = mu (Poisson)
            new_eta = np.clip(en.predict(Z), -5, 10)
            delta = np.max(np.abs(new_eta - eta))
            eta, b0, coef = new_eta, en.intercept_, en.coef_
            if delta < self.tol:
                break
        self.intercept_, self.coef_ = b0, coef
        return self

    def predict(self, X):
        return np.exp(np.clip(self.intercept_ + self._scale(X) @ self.coef_, -5, 10))


def make_linear():
    return make_pipeline(StandardScaler(), LinearRegression())


def make_pcr(n_components=None):
    return make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression())


def make_svr(C=10.0, epsilon=0.1, gamma="scale"):
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))


def make_poisson(alpha=1e-3, l1_wt=0.5):
    return PoissonElasticNet(alpha=alpha, l1_wt=l1_wt)


# name -> (factory, uses_log_target). Also used by demo/app.py
MODEL_REGISTRY = {
    "linear": (lambda: make_linear(), True),
    "poisson": (lambda: make_poisson(**BEST_PARAMS["poisson"]), False),   # exception: raw counts
    "pcr": (lambda: make_pcr(**BEST_PARAMS["pcr"]), True),
    "svr": (lambda: make_svr(**BEST_PARAMS["svr"]), True),
}

TUNE_GRIDS = {
    "pcr": (make_pcr, {"n_components": [5, 10, 20, 30, 40, None]}, True),
    "svr": (make_svr, {"C": [1, 3, 10, 30], "epsilon": [0.05, 0.1, 0.2]}, True),
    "poisson": (make_poisson, {"alpha": [1e-4, 1e-3, 1e-2], "l1_wt": [0.0, 0.5, 1.0]}, False),
}


def tune(name, X, y):
    factory, grid, log_target = TUNE_GRIDS[name]
    keys = list(grid)
    best = (np.inf, None)
    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo))
        if name == "pcr" and params["n_components"] and params["n_components"] > X.shape[1]:
            continue
        s = cv_rmsle(factory(**params), X, y, log_target)
        print(f"  {name} {params} -> {s:.4f}")
        if s < best[0]:
            best = (s, params)
    print(f"BEST {name}: {best[1]} ({best[0]:.4f})  <- copy into BEST_PARAMS")
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="linear,poisson,pcr,svr")
    ap.add_argument("--tune", action="store_true", help="grid-search pcr/svr/poisson first")
    args = ap.parse_args()

    train, test = load_data()
    X, X_test = build_features(train), build_features(test)
    y = train["count"].values

    for name in args.models.split(","):
        if args.tune and name in TUNE_GRIDS:
            BEST_PARAMS[name].update(tune(name, X, y))
        factory, log_target = MODEL_REGISTRY[name]
        model = factory()
        score = cv_rmsle(model, X, y, log_target)
        print(f"{name}: CV RMSLE = {score:.4f}")
        save_result(name, score)
        save_predictions(name, test["datetime"], fit_predict_test(model, X, y, X_test, log_target))


if __name__ == "__main__":
    main()
