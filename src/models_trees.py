"""
Person B: Tree-based models for Bike Rental Demand Forecasting.

Models:
1. Decision Tree
2. Random Forest
3. Gradient Boosting
"""

from sklearn.tree import DecisionTreeRegressor

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

    # 5-fold CV using the shared evaluation function
    score = cv_rmsle(
        model,
        X,
        y,
        log_target=True
    )

    print(f"Decision Tree CV RMSLE: {score:.4f}")

    # Save result
    save_result("decision_tree", score)


if __name__ == "__main__":
    run_decision_tree()