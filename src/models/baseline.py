"""
baseline.py

Compute baseline forecasts for Tesla 21 day returns:
  1) Naïve (last month return)
  2) Linear Regression on selected features

Usage:
    python -m src.models.baseline \
        --input_path ../../data/processed/tsla_features.csv \
        --train_fraction 0.8
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(path: str) -> pd.DataFrame:
    """Load features CSV, parse dates, and sort."""
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    return df


def build_target(df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """
    Create forward return target:  (Close_{t+horizon}/Close_t - 1).
    Drops the last `horizon` rows (no target).
    """
    df = df.copy()
    df['target_ret'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df = df.dropna(subset=['target_ret'])
    return df


def train_test_split_time(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Split df into train/test by chronological order.
    """
    split = int(len(df) * train_frac)
    train = df.iloc[:split]
    test  = df.iloc[split:]
    return train, test


def naive_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """
    Naïve forecast: predict next‐month return = past 21‐day return.
    That is, use the feature 'return_21' computed over the previous month.
    """
    # On test set, we simply copy the feature return_21 as the forecast
    return test['return_21']


def linear_regression_predict(train: pd.DataFrame, test: pd.DataFrame, features: list):
    """
    Fit a LinearRegression on `features` to predict 'target_ret'.
    Returns predictions on the test set.
    """
    X_train = train[features]
    y_train = train['target_ret']
    X_test  = test[features]

    # One‐hot encode 'month' if included
    if 'month' in features:
        X_train = pd.get_dummies(X_train, columns=['month'], drop_first=True)
        X_test  = pd.get_dummies(X_test,  columns=['month'], drop_first=True)
        # align columns in case some months missing in test/train
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return pd.Series(model.predict(X_test), index=test.index)


def evaluate(y_true: pd.Series, y_pred: pd.Series, label: str) -> dict:
    """Compute MAE and RMSE."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'model': label, 'MAE': mae, 'RMSE': rmse}


def main(args):
    # 1) Load and prepare
    df = load_data(args.input_path)
    df = build_target(df, horizon=21)

    # 2) Train/test split
    train, test = train_test_split_time(df, args.train_fraction)

    # 3) Naïve forecast
    y_true = test['target_ret']
    y_pred_naive = naive_predict(train, test)
    res_naive = evaluate(y_true, y_pred_naive, 'Naïve')

    # 4) Linear regression forecast
    #    Choose a small set of informative features:
    feat = ['return_1', 'return_5', 'rsi', 'vol_ratio_20', 'month']
    y_pred_lin = linear_regression_predict(train, test, feat)
    res_lin = evaluate(y_true, y_pred_lin, 'LinearRegression')

    # 5) Summarize
    results = pd.DataFrame([res_naive, res_lin]).set_index('model')
    print("\nBaseline Performance on Test Set:")
    print(results.round(5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline TSLA forecasting")
    parser.add_argument("--input_path",     type=str,   required=True,
                        help="Path to feature‑engineered CSV")
    parser.add_argument("--train_fraction", type=float, default=0.8,
                        help="Fraction of data to use for training")
    args = parser.parse_args()
    main(args)


"""_summary_
    Baseline forecasting for Tesla stock returns.
    
    This script implements two baseline forecasting methods:
    1. Naïve forecast using the last month's return.
    2. Linear regression using selected features.
    
    It evaluates both methods on a test set and prints performance metrics.
    
    Usage:
        python -m src.models.baseline --input_path <path_to_features_csv> --train_fraction <fraction>
        python -m src.models.baseline \
    --input_path data/processed/tsla_features.csv \
    --train_fraction 0.8
    
    ###Results:
    Baseline Performance on Test Set:
                          MAE     RMSE
    model                             
    Naïve             0.25255  0.33357
    LinearRegression  0.17464  0.22404
    
    
    1. What This Tells Us
      LinearRegression MAE ↓30% vs Naïve
      - Going from 0.2526 to 0.1746 means on average your absolute return-error dropped from ~25% to ~17.5%.
      → Our features (momentum, RSI, volume ratio, seasonality) have real predictive signal.

      RMSE also improved
      - From 0.3336 to 0.2240. RMSE penalizes large errors more heavily, so the regression model is also cutting down on the worst misses.

      Residual scale still large. Even 0.224 average error on a 21-day return is substantial (i.e. 22% off). That tells us:
      - This is a hard forecasting problem.
      - We may need more sophisticated models or additional data (macro, sentiment).
"""