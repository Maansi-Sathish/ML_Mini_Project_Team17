"""Person A: feature ablation. Tries all 2^6 = 64 on/off combinations of the six feature
ideas in src/features.py (ABLATION_FLAGS) and scores each with the same CV as the models.

Run from project root:
    python -m src.ablation            # full run (a few minutes)
    python -m src.ablation --fast     # quick test on a 2500-row sample

Outputs go to results/extra/ (NOT results/, so the model comparison is not affected):
    ablation.csv          all 64 combinations, best first
    ablation_effects.csv  average effect of each idea (ON minus OFF; negative = helps)
    ablation_effects.png  bar chart of the above
"""
import argparse
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.evaluate import cv_rmsle, load_data
from src.features import ABLATION_FLAGS, build_features

OUT = Path(__file__).resolve().parents[1] / "results" / "extra"


def make_judge():
    """Fast tree model used to judge each feature set (the paper used a tree model too)."""
    return HistGradientBoostingRegressor(max_iter=150, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="quick test on a small sample")
    args = ap.parse_args()

    train, _ = load_data()
    if args.fast:
        train = train.sample(2500, random_state=0).reset_index(drop=True)
    y = train["count"].values

    combos = list(product([False, True], repeat=len(ABLATION_FLAGS)))
    rows, t0 = [], time.perf_counter()
    for i, combo in enumerate(combos, 1):
        flags = dict(zip(ABLATION_FLAGS, combo))
        score = cv_rmsle(make_judge(), build_features(train, **flags), y, True)
        rows.append({**{k: int(v) for k, v in flags.items()}, "cv_rmsle": round(score, 5)})
        print(f"[{i:2d}/{len(combos)}] {score:.4f}  {flags}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values("cv_rmsle").reset_index(drop=True)
    effects = pd.DataFrame([{
        "idea": f,
        "mean_cv_when_ON": round(df.loc[df[f] == 1, "cv_rmsle"].mean(), 5),
        "mean_cv_when_OFF": round(df.loc[df[f] == 0, "cv_rmsle"].mean(), 5),
    } for f in ABLATION_FLAGS])
    effects["effect_ON_minus_OFF"] = (effects["mean_cv_when_ON"] - effects["mean_cv_when_OFF"]).round(5)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ablation.csv", index=False)
    effects.to_csv(OUT / "ablation_effects.csv", index=False)

    e = effects.sort_values("effect_ON_minus_OFF", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(e["idea"], e["effect_ON_minus_OFF"],
            color=["#2e9e5b" if v < 0 else "#c0392b" for v in e["effect_ON_minus_OFF"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Change in CV RMSLE when the idea is ON (negative = helps)")
    ax.set_title("Feature ablation: average effect of each idea")
    fig.tight_layout()
    fig.savefig(OUT / "ablation_effects.png", dpi=150)
    plt.close(fig)

    print("\nTop 5 combinations:")
    print(df.head(5).to_string(index=False))
    print("\nAverage effect of each idea (negative = helps):")
    print(effects.sort_values("effect_ON_minus_OFF").to_string(index=False))
    print(f"\nSaved to {OUT}")
    print("Next: set the best combination as the defaults of build_features() in src/features.py")


if __name__ == "__main__":
    main()