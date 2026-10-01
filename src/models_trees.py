"""
Person B: Tree-based models for Bike Rental Demand Forecasting.

Models:
1. Decision Tree
2. Random Forest
3. Gradient Boosting
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.evaluate import load_data, cv_rmsle, save_result
from src.features import build_features


def run_decision_tree():
    # Load data
    train, test = load_data()

    # Build features
    X = build_features(train)
    y = train["count"]

    # Decision Tree
    model = DecisionTreeRegressor(
        random_state=42
    )

    score = cv_rmsle(
        model,
        X,
        y,
        log_target=True
    )

    print(f"Decision Tree CV RMSLE: {score:.4f}")

    save_result("decision_tree", score)


def run_random_forest():
    # Load data
    train, test = load_data()

    # Build features
    X = build_features(train)
    y = train["count"]

    # Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    score = cv_rmsle(
        model,
        X,
        y,
        log_target=True
    )

    print(f"Random Forest CV RMSLE: {score:.4f}")

    save_result("random_forest", score)


if __name__ == "__main__":
    run_decision_tree()
    run_random_forest()