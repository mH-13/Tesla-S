# Tesla Stock Forecasting Project

A complete end‑to‑end pipeline for forecasting Tesla (TSLA) monthly returns using a variety of machine learning, statistical, and deep learning models. This README guides you through setup, data ingestion, feature engineering, exploratory analysis, model training, evaluation, ensembling, and next‑steps.

---

## Table of Contents

- [Tesla Stock Forecasting Project](#tesla-stock-forecasting-project)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Directory Structure](#directory-structure)
  - [Prerequisites \& Installation](#prerequisites--installation)
  - [Data Pipeline](#data-pipeline)
    - [1. Data Ingestion](#1-data-ingestion)
    - [2. Feature Engineering](#2-feature-engineering)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Modeling \& Evaluation](#modeling--evaluation)
    - [Baselines](#baselines)
    - [Cross‑Validation \& Walk‑Forward](#crossvalidation--walkforward)
    - [SARIMA](#sarima)
    - [LSTM](#lstm)
    - [Hyperparameter Tuning (XGBoost)](#hyperparameter-tuning-xgboost)
    - [Regime Flag \& Stacking Ensemble](#regime-flag--stacking-ensemble)
    - [Metrics \& Awards](#metrics--awards)
  - [Results Summary](#results-summary)
  - [Visualization Highlights](#visualization-highlights)
  - [Further Experiment(Optional)](#further-experimentoptional)

---

## Project Overview

We aim to forecast Tesla’s 21‑day forward return using historical price and volume data. The pipeline:

1. **Ingests and cleans** raw CSV data
2. **Engineers features** (returns, moving averages, volatility, RSI, calendar)
3. **Performs EDA** to understand distributions and relationships
4. **Trains multiple models**: Naïve, Linear Regression, Random Forest, XGBoost, SARIMA, LSTM
5. **Hyperparameter tunes** XGBoost via randomized search
6. **Adds regime detection** (pre‑/post‑2020 structural shift)
7. **Stacks** XGBoost, LSTM, and SARIMA with a Ridge meta‑learner
8. **Evaluates** via MAE, RMSE, R², Directional Accuracy
9. **Visualizes** performance and awards the best models

---

## Directory Structure

```text
tesla-forecast/
├── data/
│   ├── raw/                   # Original TSLA.csv
│   └── processed/             # Cleaned and feature CSVs
│       ├── tsla_cleaned.csv
│       └── tsla_features.csv
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_modeling_and_evaluation.ipynb
│       └── Sections:
│           • Environment Setup
│           • Data Load & Target
│           • Baselines & CV
│           • SARIMA, LSTM
│           • Hyperparameter Tuning
│           • Regime & Stacking
│           • Awards & Visuals
├── src/
│   ├── data/
│   │   └── make_dataset.py    # Ingest & clean raw CSV
│   └── features/
│       └── build_features.py  # Compute engineered features
├── models/                    # (Optional) saved model pickles
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Prerequisites & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/mH-13/Tesla-S
   cd Tesla-S
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Data Pipeline

### 1. Data Ingestion

Clean raw TSLA CSV, parse dates, drop duplicates & missing values, and index by `Date`.

```bash
python -m src.data.make_dataset \
  --input_path data/raw/TSLA.csv \
  --output_path data/processed/tsla_cleaned.csv
```

### 2. Feature Engineering

Compute lagged returns, moving averages, volatility, RSI, volume ratios, and calendar features.

```bash
python -m src.features.build_features \
  --input_path data/processed/tsla_cleaned.csv \
  --output_path data/processed/tsla_features.csv
```

---

## Exploratory Data Analysis

Open and run `notebooks/01_eda.ipynb` to:

* View summary statistics
* Plot time series of Close price and moving averages
* Visualize return distributions and fat tails
* Correlation heatmap of features
* Rolling volatility and seasonal decomposition

---

## Modeling & Evaluation

Open and run `notebooks/02_modeling_and_evaluation.ipynb` which covers:

### Baselines

* **Naïve**: last 21‑day return
* **Linear Regression**, **Random Forest**, **XGBoost** on a core feature subset

### Cross‑Validation & Walk‑Forward

* Chronological 80/20 split
* `TimeSeriesSplit` for robust walk‑forward validation

### SARIMA

* Seasonal ARIMA(1,1,1)(1,1,1,12) on monthly returns
* Index alignment fix to avoid label mismatch errors

### LSTM

* 30‑day sliding windows of key features
* Single‑layer LSTM → Dense(1)

### Hyperparameter Tuning (XGBoost)

* `RandomizedSearchCV` over depth, learning rate, subsample, n\_estimators
* Best MAE→0.14, RMSE→0.18, R²→0.31, DirAcc→0.63

### Regime Flag & Stacking Ensemble

* **Regime**: binary indicator (pre‑2020 vs post‑2020)
* **Stack**: Tuned XGBoost, LSTM, SARIMA meta‑features → Ridge meta‑learner
* Highest directional accuracy (0.66) with MAE/RMSE near best

### Metrics & Awards

* Compiled into a DataFrame with MAE, RMSE, R², Directional Accuracy
* Automated “🏆”, “🏅”, “⭐” awards for best MAE, DirAcc, R²

---

## Results Summary

| Model             | MAE    | RMSE   | R²      | DirAcc | Award               |
| ----------------- | ------ | ------ | ------- | ------ | ------------------- |
| Naïve             | 0.2526 | 0.3336 | –1.3289 | 0.4564 |                     |
| Linear Regression | 0.1746 | 0.2240 | –0.0505 | 0.5257 |                     |
| Random Forest     | 0.1787 | 0.2300 | –0.1074 | 0.5034 |                     |
| XGBoost           | 0.1886 | 0.2407 | –0.2127 | 0.5034 |                     |
| LSTM              | 0.2040 | 0.2801 | –0.6518 | 0.4694 |                     |
| XGB Tuned         | 0.1403 | 0.1810 | 0.3141  | 0.6331 | ⭐ Best R²           |
| XGB Walk‑Forward  | 0.0991 | 0.1258 | –0.5371 | 0.5296 | 🏆 Best MAE         |
| SARIMA            | 0.2139 | 0.2471 | –0.5453 | 0.3636 |                     |
| Stack Ensemble    | 0.1101 | 0.1408 | 0.3055  | 0.6555 | 🏅 Best Directional |

---

## Visualization Highlights

* **MAE vs Directional Accuracy Scatter**
* **Bar charts** for MAE/RMSE comparison
* **Dual‑axis plots** for R² vs DirAcc

Run the final cells in `02_modeling_and_evaluation.ipynb` to generate these visuals.

---

## Further Experiment(Optional)

1. **Transformer Models**: e.g. Autoformer or Informer for long‑range dependencies
2. **Advanced Ensembling**: Bayesian model averaging or multi‑layer stacking
3. **Feature Enrichment**: sentiment data, macro indicators, alternative sources
4. **Probabilistic Forecasting**: quantile regression or deep ensembles
5. **CI/CD & Testing**: add `pytest` tests to lock in key metric thresholds
