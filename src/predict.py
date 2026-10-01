"""
Person B: Generate final test-set predictions
using tuned tree-based models.
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from src.evaluate import (
    load_data,
    fit_predict_test,
    save_predictions
)

from src.features import build_features


def generate_predictions():

    train, test = load_data()

    X_train = build_features(train)
    X_test = build_features(test)

    y = train["count"]

    models = {

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

    for name, model in models.items():

        predictions = fit_predict_test(
            model,
            X_train,
            y,
            X_test
        )

        save_predictions(
            name,
            test["datetime"],
            predictions
        )

        print(
            f"{name} predictions saved successfully."
        )


if __name__ == "__main__":
    generate_predictions()