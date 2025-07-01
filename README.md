# Tesla‑S: Tesla Stock Price Forecasting

## Project Overview
This repository implements a full pipeline to forecast month‑ahead closing prices (or returns) for Tesla (TSLA) stock. It covers:

1. **Data Ingestion & Cleaning**  
2. **Feature Engineering**  
3. **Exploratory Data Analysis (EDA)**  
4. **Baseline & Advanced Modeling**  
5. **Evaluation & Visualization**

Along the way you’ll see how each step is motivated by our data insights and forecasting goals.

---

## 📂 Directory Structure
```

Tesla‑S/
├── data/
│   ├── raw/              # Original TSLA.csv (Kaggle download)
│   └── processed/        # Cleaned & feature‑engineered CSVs
│       ├── tsla\_cleaned.csv
│       └── tsla\_features.csv
├── notebooks/
│   ├── 01\_eda.ipynb      # Visual data exploration
│   └── 02\_modeling\_and\_evaluation.ipynb
│                          # Baselines, tree‑models, SARIMA, LSTM
├── src/
│   ├── data/
│   │   └── make\_dataset.py        # Ingest & clean raw CSV
│   ├── features/
│   │   └── build\_features.py      # Compute returns, MAs, RSI, volatility, etc.
│   └── models/
│       └── baseline.py            # Naïve & LinearRegression baselines
├── tests/                # (Future) unit tests for each module
├── requirements.txt      # Pinned Python dependencies
└── README.md             # This file

```

---

## Setup Instructions

1. **Clone the repo**  
```bash
   git clone <https://github.com/mH-13/Tesla-S>
   cd Tesla‑S
```

2. **Create & activate** a virtual environment

```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. **Install dependencies**

```bash
   pip install -r requirements.txt
```

4. **Download** the Tesla CSV (e.g. from Kaggle) and place as `data/raw/TSLA.csv`.

---



## 1. Data Ingestion & Cleaning

* **Script**: `src/data/make_dataset.py`
* **Actions**:

  1. Load raw CSV with `pd.read_csv()`
  2. Parse dates, drop invalid/missing rows
  3. Sort by date, drop duplicates
  4. Drop rows with any missing OHLCV
  5. Set `Date` as index
* **Output**: `data/processed/tsla_cleaned.csv`

```bash
python -m src.data.make_dataset \
    --input_path data/raw/TSLA.csv \
    --output_path data/processed/tsla_cleaned.csv
```

---

## ⚙️ 2. Feature Engineering

* **Script**: `src/features/build_features.py`
* **Features Created**:

  * Lagged returns: 1, 5, 21 days
  * Moving averages: 5, 10, 20 days
  * Volatility (10‑day rolling std of returns)
  * RSI (14‑day)
  * Volume features: 20‑day rolling average & ratio
  * Calendar: day‑of‑week, month, quarter
* **Cleaning**: Drop any rows with NaNs from rolling calculations
* **Output**: `data/processed/tsla_features.csv`

```bash
python -m src.features.build_features \
    --input_path data/processed/tsla_cleaned.csv \
    --output_path data/processed/tsla_features.csv
```

---



## 3. Exploratory Data Analysis (EDA)

* **Notebook**: `notebooks/01_eda.ipynb`
* **Key Plots & Insights**:

  * **Price & MA**: trend smoothing, bull/bear phases
  * **Returns Distribution**: fat tails, outliers ±20%
  * **Volatility Over Time**: spikes during crises
  * **Correlation Heatmap**: high collinearity among price/MA, stronger 21‑day return ↔ RSI
  * **Volume vs Returns**: spikes align with extreme moves
  * **Seasonal Decomposition**: monthly seasonality & heteroskedasticity
* **Decision Framework**:

  * Drop redundant features for linear models
  * Frame target as 21‑day return
  * Use robust metrics (MAE) and direction accuracy

---



## 📊 4. Modeling & Evaluation

* **Notebook**: `notebooks/02_modeling_and_evaluation.ipynb`

* **Models Trained**:

  1. **Naïve**: predict next‑month = last‑month (`return_21`)
  2. **Linear Regression** on selected features
  3. **Random Forest**
  4. **XGBoost** (single split + walk‑forward validation)
  5. **SARIMA** (seasonal ARIMA on monthly series)
  6. **LSTM** (30‑day sliding window)

* **Evaluation Metrics**:

  * **MAE**: mean absolute error (robust to outliers)
  * **RMSE**: root mean squared error (penalizes large misses)
  * **MAPE**: mean absolute percentage error (caution near zero returns)
  * **R²**: variance explained (negative indicates worse than mean)
  * **Directional Accuracy**: % correct sign predictions

* **Consolidated Results**:

  | Model            | MAE    | RMSE   | MAPE   | R²      | DirAcc |
  | ---------------- | ------ | ------ | ------ | ------- | ------ |
  | Naïve            | 0.2526 | 0.3336 | 4.4813 | –1.3289 | 45.6%  |
  | LinearRegression | 0.1746 | 0.2240 | 1.9051 | –0.0505 | 52.6%  |
  | RandomForest     | 0.1807 | 0.2360 | 1.7758 | –0.1654 | 51.0%  |
  | XGBoost          | 0.1826 | 0.2335 | 1.9335 | –0.1411 | 48.1%  |
  | XGBoost\_WFV     | 0.1647 | 0.2078 | 2.2557 | –0.4223 | 51.3%  |
  | SARIMA           | 0.1906 | 0.2315 | 1.7558 | –0.3793 | 40.0%  |
  | LSTM             | 0.1766 | 0.2429 | 1.3192 | –0.2420 | 47.6%  |

---



## Next Steps

1. **Hyperparameter Tuning**

   * Tree‑models (XGBoost, RF) with `TimeSeriesSplit`
   * Regularized linear models (Ridge, Lasso)
   * SARIMAX order search (AIC/BIC)
   * LSTM architecture & training tweaks

2. **Advanced Features**

   * Regime flags (high vs low volatility)
   * Exogenous data (VIX, S\&P 500, sentiment)
   * Additional technical indicators (MACD, Bollinger Bands)

3. **Ensembling & Stacking**

   * Blend best models via simple averaging or meta‑learners

4. **Walk‑Forward Backtest & Deployment**

   * Automate rolling retrain & forecast
   * Simulate trading strategy (Sharpe, drawdown)
