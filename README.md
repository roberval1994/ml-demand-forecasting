<h1 align="center">🚲 Demand Forecasting — End-to-End Machine Learning Project</h1>

<p align="center">
  <b>A complete, documented ML pipeline for hourly demand forecasting</b><br>
  Data ingestion → EDA → feature engineering → modelling → tuning → evaluation → explainability
</p>

🌐 **Language / Idioma:** **English** | [Português](README.pt-BR.md)

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style](https://img.shields.io/badge/style-PEP8-informational.svg)]()

## 🎯 Goal

This repository is a **reference end-to-end Machine Learning project** built to be read,
reused, and extended. It forecasts **hourly demand** (bike-sharing rentals) and
demonstrates every stage of a professional ML workflow with clean, modular, tested code.

It doubles as my personal **study and consultation template** for future projects.

## 🧭 Why bike-sharing demand?

The [Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
(UCI) is a public, tabular **time-series regression** problem with rich structure:
seasonality (hour, weekday, season), weather effects, and holidays. It is ideal to
showcase feature engineering and model comparison without any licensing concerns.

## 🗂️ Project structure

```
ml-demand-forecasting/
├── data/
│   ├── raw/                     # Original dataset (downloaded, not versioned)
│   └── processed/               # Cleaned & feature-engineered data (generated)
├── notebooks/                   # (optional) exploratory notebooks
├── src/demand_forecasting/
│   ├── __init__.py
│   ├── config.py                # Central configuration (paths, params)
│   ├── data.py                  # Download & load the dataset
│   ├── features.py              # Feature engineering (calendar, cyclical, lags)
│   ├── model.py                 # Model factory & training pipeline
│   ├── evaluate.py              # Metrics & diagnostic plots
│   └── explain.py               # Feature importance / SHAP explainability
├── scripts/
│   └── run_pipeline.py          # One-command end-to-end run
├── tests/
│   ├── test_features.py
│   └── test_data.py
├── requirements.txt
├── LICENSE
├── README.md                    # English (this file)
└── README.pt-BR.md              # Portuguese
```

## 🔬 Pipeline stages

| Stage | Module | What it does |
|---|---|---|
| **1. Data** | `data.py` | Downloads the UCI dataset and loads it into a tidy DataFrame |
| **2. EDA** | `notebooks/` | Distributions, seasonality, correlations |
| **3. Features** | `features.py` | Calendar features, **cyclical encoding** (sin/cos), lag & rolling features |
| **4. Model** | `model.py` | Baseline + Random Forest + Gradient Boosting in a scikit-learn `Pipeline` (+ **optional XGBoost**) |
| **5. Tuning** | `model.py` | Cross-validated hyperparameter search with time-aware splits |
| **6. Evaluation** | `evaluate.py` | MAE, RMSE, R²; residual & prediction plots |
| **7. Explainability** | `explain.py` | Permutation importance and (optional) SHAP values |

## ⚡ Quick start

```bash
# 1. Clone
git clone https://github.com/roberval1994/ml-demand-forecasting.git
cd ml-demand-forecasting

# 2. Environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt

# 3. Run the whole pipeline end-to-end
python scripts/run_pipeline.py

# 4. (optional) run the tests
pytest -q
```

The pipeline downloads the data, builds features, trains and tunes the models, evaluates
them on a hold-out test set, and writes metrics and plots to `outputs/`.

## 📈 What the pipeline reports

- A **leaderboard** comparing baseline vs tree-based models (MAE / RMSE / R²).
- **Residual** and **actual-vs-predicted** diagnostic plots.
- **Feature importance** ranking to interpret the drivers of demand.

## 🧪 Tests

```bash
pytest -q
```

Unit tests cover the feature-engineering logic and the data loader to guard against
regressions — a habit worth keeping in any serious ML codebase.

## 🧰 Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `seaborn` · `SHAP` · `pytest`

**Optional extra:** `XGBoost` unlocks Section 8 of the walkthrough notebook (advanced
boosting model). The pipeline runs fine without it; install with `pip install xgboost`.

## 📚 Design principles

- **Modular**: logic lives in `src/`, notebooks only orchestrate and narrate.
- **Reproducible**: fixed seeds, pinned dependencies, one-command run.
- **Time-aware**: validation respects temporal order (no leakage from the future).
- **Documented**: every module has a docstring explaining the "why", not just the "how".

## 👤 Author

**Roberval Gonçalves Moreira Filho** — Data Scientist | Operational Research Analyst

[![LinkedIn](https://img.shields.io/badge/LinkedIn-robervalOr-blue)](https://www.linkedin.com/in/robervalOr)
[![Lattes](https://img.shields.io/badge/Lattes-CNPq-00599C)](http://lattes.cnpq.br/4394523940603239)
[![GitHub](https://img.shields.io/badge/GitHub-roberval1994-black)](https://github.com/roberval1994)

## 📄 License

Released under the MIT License. See [LICENSE](LICENSE).
