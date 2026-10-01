"""
Main pipeline for Bike Rental Demand Forecasting.

Runs:
1. Tree-based model evaluation
2. Final test-set prediction generation
"""

from src.models_trees import (
    run_decision_tree,
    run_random_forest,
    run_gradient_boosting
)

from src.predict import generate_predictions


def main():
    print("=" * 60)
    print("BIKE RENTAL DEMAND FORECASTING")
    print("=" * 60)

    print("\n[1] Running tree-based models...")
    
    run_decision_tree()
    run_random_forest()
    run_gradient_boosting()

    print("\n[2] Generating final test-set predictions...")

    generate_predictions()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()