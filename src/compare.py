"""Final comparison of all models (Person A).

Reads every results/<name>.csv that is a score file (columns model,cv_rmsle, one row,
model == file name), so extra files such as *_tuning.csv, error_by_*.csv and
final_model_comparison.csv are ignored. Writes:
    results/summary/model_comparison.csv
    results/summary/model_comparison.png
Run on its own:  python -m src.compare
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # save to file, no window needed
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TREE_MODELS = {"decision_tree", "random_forest", "gradient_boosting", "stacking_ensemble"}
LINEAR_MODELS = {"linear", "poisson", "pcr", "svr"}


def collect_scores(results_dir=RESULTS):
    rows = []
    for f in sorted(Path(results_dir).glob("*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if list(df.columns) == ["model", "cv_rmsle"] and len(df) == 1 and df["model"].iloc[0] == f.stem:
            rows.append((f.stem, float(df["cv_rmsle"].iloc[0])))
    return pd.DataFrame(rows, columns=["model", "cv_rmsle"]).sort_values("cv_rmsle").reset_index(drop=True)


def compare_models(results_dir=RESULTS):
    table = collect_scores(results_dir)
    out_dir = Path(results_dir) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "model_comparison.csv", index=False)

    def colour(m):
        return "#2a6fbb" if m in TREE_MODELS else "#e08a1e" if m in LINEAR_MODELS else "#888888"

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(table) + 1.8))
    ax.barh(table["model"], table["cv_rmsle"], color=[colour(m) for m in table["model"]])
    ax.invert_yaxis()                                   # best model on top
    for i, v in enumerate(table["cv_rmsle"]):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("5-fold CV RMSLE (lower is better)")
    ax.set_title("Bike demand: model comparison")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#2a6fbb"), plt.Rectangle((0, 0), 1, 1, color="#e08a1e")],
              labels=["Tree-based", "Linear / kernel"], loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    print(table.round(4).to_string(index=False))
    print(f"\nSaved: {out_dir / 'model_comparison.csv'}\n       {out_dir / 'model_comparison.png'}")
    return table


if __name__ == "__main__":
    compare_models()