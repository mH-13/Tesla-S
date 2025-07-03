"""
build_features.py

Take cleaned Tesla data and compute engineered features for modeling.

Usage:
    python -m src.features.build_features \
        --input_path ../../data/processed/tsla_cleaned.csv \
        --output_path ../../data/processed/tsla_features.csv
"""

import argparse
import pandas as pd
import numpy as np


def compute_lagged_returns(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Compute percentage returns for given lag windows.

    Args:
        df: Date‑indexed DataFrame with 'Close' column.
        windows: List of integer windows (in trading days).
    Returns:
        df with new columns 'return_X' for each window X.
    """
    for w in windows:
        col_name = f"return_{w}"
        df[col_name] = df['Close'].pct_change(periods=w)
    return df


def compute_moving_averages(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Compute rolling moving averages of 'Close'.

    Args:
        df: Date‑indexed DataFrame with 'Close'.
        windows: List of integer windows.
    Returns:
        df with new columns 'ma_X'.
    """
    for w in windows:
        df[f"ma_{w}"] = df['Close'].rolling(window=w).mean()
    return df


def compute_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling volatility (std dev) of daily returns.

    Args:
        df: Date‑indexed DataFrame with 'Close'.
        window: Window size for std deviation.
    Returns:
        df with 'volatility' column.
    """
    daily_ret = df['Close'].pct_change()
    df[f"vol_{window}"] = daily_ret.rolling(window=window).std()
    return df


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).

    Args:
        df: Date‑indexed DataFrame with 'Close'.
        window: Window size for RSI (default 14).
    Returns:
        df with 'rsi' column.
    """
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use simple moving average of gains/losses
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def compute_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute volume-based features:
    - Rolling average volume
    - Volume / rolling avg volume

    Args:
        df: Date‑indexed DataFrame with 'Volume'.
        window: Window size for rolling average.
    Returns:
        df with 'vol_ma', 'vol_ratio'.
    """
    df[f"vol_ma_{window}"] = df['Volume'].rolling(window=window).mean()
    df[f"vol_ratio_{window}"] = df['Volume'] / df[f"vol_ma_{window}"]
    return df


def compute_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar features from the Date index.

    Args:
        df: Date‑indexed DataFrame.
    Returns:
        df with 'day_of_week', 'month', 'quarter'.
    """
    df['day_of_week'] = df.index.dayofweek    # Monday=0 … Friday=4
    df['month']       = df.index.month
    df['quarter']     = df.index.quarter
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate all feature computations.

    Args:
        df: Cleaned, Date‑indexed DataFrame.
    Returns:
        DataFrame with original columns plus new features.
    """
    df = compute_lagged_returns(df, windows=[1, 5, 21])
    df = compute_moving_averages(df, windows=[5, 10, 20])
    df = compute_volatility(df, window=10)
    df = compute_rsi(df, window=14)
    df = compute_volume_features(df, window=20)
    df = compute_date_features(df)

    # After all, drop rows with NaNs introduced by rolling operations
    df = df.dropna()
    return df


def main(args):
    # 1. Load cleaned data
    df = pd.read_csv(args.input_path, index_col='Date', parse_dates=True)
    print(f"[build_features] Loaded {len(df)} rows")

    # 2. Build features
    df_feat = build_features(df)
    print(f"[build_features] After feature engineering: {len(df_feat)} rows, {df_feat.shape[1]} columns")

    # 3. Save to CSV
    df_feat.to_csv(args.output_path)
    print(f"[build_features] Features written to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute TSLA features")
    parser.add_argument("--input_path",  type=str, required=True, help="Path to cleaned CSV")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save features CSV")
    args = parser.parse_args()
    main(args)


"""python -m src.features.build_features \
    --input_path data/processed/tsla_cleaned.csv \
    --output_path data/processed/tsla_features.csv


head -n 5 data/processed/tsla_features.csv"""