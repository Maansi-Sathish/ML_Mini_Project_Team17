"""
Main pipeline for Bike Rental Demand Forecasting.

Runs:
1. Tree-based model evaluation
2. Linear / kernel models (linear, Poisson GLM, PCR, SVR)
3. Final test-set prediction generation
4. Final comparison table and bar chart of all models (results/summary/)

Usage:
    python main.py                # run everything
    python main.py --skip-linear  # skip step 2 (reuses results/ and predictions/ already saved)
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.models_trees import (
    run_decision_tree,
    run_random_forest,
    run_gradient_boosting
)

from src.predict import generate_predictions
from src.compare import compare_models

ROOT = Path(__file__).resolve().parent


def run_linear_models():
    """Runs src/models_linear.py (Person A) as a separate process."""
    subprocess.run([sys.executable, "-m", "src.models_linear"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-linear", action="store_true",
                        help="skip the linear/kernel models (SVR etc. take a few minutes)")
    args = parser.parse_args()

    print("=" * 60)
    print("BIKE RENTAL DEMAND FORECASTING")
    print("=" * 60)

    print("\n[1] Running tree-based models...")

    run_decision_tree()
    run_random_forest()
    run_gradient_boosting()

    if args.skip_linear:
        print("\n[2] Skipping linear/kernel models (--skip-linear)")
    else:
        print("\n[2] Running linear/kernel models (linear, Poisson GLM, PCR, SVR)...")
        run_linear_models()

    print("\n[3] Generating final test-set predictions...")

    generate_predictions()

    print("\n[4] Comparing all models...")

    compare_models()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()