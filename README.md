# Forecasting Bike Rental Demand

UE24CS352A Machine Learning - Mini-Project (Team 17)

Predict the **hourly number of bike rentals** for the Capital Bikeshare system in Washington D.C. from date, hour and weather information. This is the Kaggle *Bike Sharing Demand* problem. The evaluation metric is **RMSLE** (Root Mean Squared Logarithmic Error); lower is better.

## Team

| Member | GitHub | Part |
|---|---|---|
| Maansi S (Person A)| Maansi-Sathish | EDA, feature engineering, feature ablation, linear / kernel models, casual-vs-registered experiment, model comparison, Streamlit demo |
| Amar Navdogi (Person B) | Amar1919 | Tree-based models, tuning, stacking ensemble, error analysis, prediction pipeline |

## Dataset

- Source: Kaggle "Bike Sharing Demand" (https://www.kaggle.com/c/bike-sharing-demand). Both files are in `data/`.
- `data/train.csv`: 10,886 hourly rows (first 19-20 days of each month, 2011-2012) with the target `count` (= `casual` + `registered`).
- `data/test.csv`: 6,493 hourly rows (remaining days of each month), target withheld.
- Features: `datetime`, `season`, `holiday`, `workingday`, `weather`, `temp`, `atemp`, `humidity`, `windspeed`.

## Approach (short)

- **Target:** every model is trained on `log1p(count)` and predictions are converted back with `expm1` (the Poisson GLM is the exception and uses the raw count). With this, plain RMSE on the log target equals RMSLE.
- **Validation:** the same 5-fold cross-validation for every model (`src/evaluate.py`).
- **Features** (`src/features.py`): hour, year, month, day of week, weekday/weekend peak-hour flags, one-hot encoding and temperature buckets for linear models, holiday dropped. Six on/off feature ideas were tested with a feature ablation (`src/ablation.py`).
- **Models:** Linear regression, Poisson GLM with elastic-net penalty, PCR, SVR (RBF) | Decision Tree, Random Forest, Gradient Boosting, stacking ensemble.

## Setup

Tested with Python 3.13.

```bash
git clone https://github.com/Maansi-Sathish/ML_Mini_Project_Team17.git
cd ML_Mini_Project_Team17
python -m venv venv                 # optional but recommended
venv\Scripts\activate               # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

All commands below are run **from the project root folder**.

## How to run

### Everything (full pipeline)

```bash
python main.py
```

This runs the tree models, then the linear/kernel models, then generates the test-set predictions and finally the model comparison table and chart. It takes a few minutes (SVR alone is about 3 minutes).

```bash
python main.py --skip-linear        # skip the linear/kernel models and reuse the results already saved
```

### Person A parts

```bash
python -m src.models_linear                      # linear, Poisson GLM, PCR, SVR
python -m src.models_linear --models svr,pcr     # only some models
python -m src.models_linear --tune               # grid search first (slower)
python -m src.ablation                           # feature ablation: all 64 combinations (about 6 minutes)
python -m src.casual_registered                  # casual + registered vs total-count experiment
python -m src.compare                            # comparison table and bar chart of all models
python -m streamlit run demo/app.py              # interactive demo (opens in the browser)
```

- The demo trains an SVR from `data/train.csv` the first time it opens (takes a short while), then answers instantly. Pick a date, hour and weather and it shows the predicted rentals and the predicted curve across the day.
- EDA: open `notebooks/eda_A.ipynb` (Jupyter or VS Code) and run all cells. It reads `../data/`, so keep the notebook in the `notebooks/` folder.

### Person B parts

```bash
python main.py                      # runs the tree models (step 1) and generates the test-set predictions (step 3)
```

Run a single tree model on its own (from the project root):

```bash
python -c "from src.models_trees import run_random_forest; run_random_forest()"
```

The same works for `run_decision_tree` and `run_gradient_boosting`.

- `src/models_trees.py`: Decision Tree, Random Forest and Gradient Boosting.
- `src/ensemble.py`: stacking ensemble (its score is in `results/stacking_ensemble.csv`).
- `src/predict.py`: generates the final test-set predictions into `predictions/`.
- `notebooks/tuning_B.ipynb`: hyperparameter tuning, feature importance and error analysis (open in Jupyter or VS Code and run all cells). Its outputs are saved in `results/` (`*_tuning.csv`, `random_forest_feature_importance.csv`, `error_by_hour.csv`, `error_by_weather.csv`, `error_by_workingday.csv`).

## Project structure

```
data/                  train.csv, test.csv
src/
  features.py          build_features(df, **flags)           (A)
  evaluate.py          RMSLE, shared 5-fold CV, saving       (shared)
  models_linear.py     Linear, Poisson GLM, PCR, SVR         (A)
  ablation.py          feature ablation (64 combinations)    (A)
  casual_registered.py casual vs registered experiment       (A)
  compare.py           final comparison table and chart      (A)
  models_trees.py      Decision Tree, Random Forest, GBM     (B)
  ensemble.py          stacking ensemble                     (B)
  predict.py           final test-set predictions            (B)
notebooks/             eda_A.ipynb (A), tuning_B.ipynb (B)
demo/app.py            Streamlit demo                        (A)
results/               one <model>.csv per model (cv_rmsle); results/extra/ and results/summary/ hold the experiments and the comparison
predictions/           test-set predictions per model (datetime,count)
main.py                runs the full pipeline
```

## Results (5-fold CV RMSLE, lower is better)

| Model | CV RMSLE |
|---|---|
| Gradient Boosting | 0.3063 |
| Random Forest | 0.3214 |
| SVR (RBF) | 0.3220 |
| Decision Tree | 0.4105 |
| Linear regression | 0.5076 |
| PCR (all components) | 0.5076 |
| Poisson GLM (elastic net) | 0.5530 |

The stacking ensemble result and the comparison chart are in `results/summary/` after running `python main.py`.

Experiments (`results/extra/`):

- **Feature ablation** (fast tree model as judge): day of week helped most, then month and the peak-hour flags; one-hot encoding hurts tree models but is needed for linear models. Best combination scored 0.2826.
- **Casual vs registered:** predicting casual and registered users separately and adding them (0.2784) was slightly better than one model on total count (0.2828), but the gap is small compared with the fold-to-fold spread.

## Notes and limitations

- Cross-validation shuffles rows randomly, so rows from the same month appear in both training and validation folds. The real test set is the later days of each month, so the CV scores are probably optimistic compared with the Kaggle leaderboard.
- Re-running `main.py` regenerates the files in `predictions/` and `results/`; some tree models are random, so small differences between runs are normal.