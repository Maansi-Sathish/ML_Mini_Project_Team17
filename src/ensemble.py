"""
Person B: Stacking ensemble for Bike Rental Demand Forecasting.

Base models:
1. Decision Tree
2. Random Forest
3. Gradient Boosting

The ensemble uses out-of-fold predictions to avoid data leakage.
"""

import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from src.evaluate import load_data, rmsle, save_result
from src.features import build_features


SEED = 42
N_SPLITS = 5


def get_models():
    """Return the tuned base models."""

    return {
        "decision_tree": DecisionTreeRegressor(
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=5,
            random_state=42
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=42,
            n_jobs=-1
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }


def generate_oof_predictions(models, X, y):
    """
    Generate out-of-fold predictions for each base model.

    Base models are trained on log1p(count).
    Predictions are converted back to count scale
    using expm1().
    """

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    oof_predictions = np.zeros(
        (len(X), len(models))
    )

    y_log = np.log1p(y)

    for model_index, (model_name, model) in enumerate(models.items()):

        print(f"Generating OOF predictions for {model_name}...")

        for train_idx, valid_idx in kfold.split(X):

            fold_model = clone(model)

            fold_model.fit(
                X.iloc[train_idx],
                y_log.iloc[train_idx]
            )

            log_pred = fold_model.predict(
                X.iloc[valid_idx]
            )

            count_pred = np.expm1(log_pred)

            oof_predictions[
                valid_idx,
                model_index
            ] = count_pred

    return oof_predictions


def run_stacking():
    """Train and evaluate the stacking ensemble."""

    train, test = load_data()

    X = build_features(train)
    y = train["count"]

    models = get_models()

    # Generate leakage-free OOF predictions
    oof_predictions = generate_oof_predictions(
        models,
        X,
        y
    )

    # Meta-model operates on count-scale predictions
    meta_model = LinearRegression()

    meta_model.fit(
        oof_predictions,
        y
    )

    # Generate ensemble predictions
    ensemble_predictions = meta_model.predict(
        oof_predictions
    )

    # Evaluate using RMSLE
    score = rmsle(
        y,
        ensemble_predictions
    )

    print(
        f"Stacking Ensemble CV RMSLE: {score:.4f}"
    )

    # Save result
    save_result(
        "stacking_ensemble",
        score
    )

    return score


if __name__ == "__main__":
    run_stacking()