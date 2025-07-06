""" This file take cleaned Tesla data and compute engineered features for modeling.
Run:
    python -m src.features.build_features \
        --input_path data/processed/tsla_cleaned.csv \
        --output_path data/processed/tsla_features.csv
"""

import argparse
import pandas as pd
import numpy as np

#lagged returns: is the percentage change in the stock price over a specified number of trading days.
# This is useful for capturing momentum or mean-reversion effects in stock prices.

def compute_lagged_returns(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    
    """This compute percentage returns for given lag windows.
    Args    : Date‑indexed DataFrame with 'Close' column. And list of integer windows (in trading days).
    Output  : df with new columns 'return_X' for each window X. """
    
    for w in windows:
        col_name = f"return_{w}"
        df[col_name] = df['Close'].pct_change(periods=w)
    return df

# Moving averages: smooths out fluctuations in data by averaging data points over a moving window
#used to identify trends by reducing short-term noise and highlighting longer-term patterns in a dataset
def compute_moving_averages(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    
    """Compute rolling moving averages of 'Close'.
    Args    : Same as compute_lagged_returns.
    Output  : df with new columns 'ma_X'. """
    
    for w in windows:
        df[f"ma_{w}"] = df['Close'].rolling(window=w).mean()
    return df

# Volatility: measures the degree of variation of a trading price series over time (standard deviation of returns).It is used to assess the risk associated with a security or market. Higher volatility - higher risk, lower volatility - more stable prices.
def compute_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    
    """ Compute rolling volatility (std dev) of daily returns.
    Args    : df: Date‑indexed DataFrame with 'Close'. window: Window size for std deviation.
    output  : df with 'volatility' column."""
    daily_ret = df['Close'].pct_change()
    df[f"vol_{window}"] = daily_ret.rolling(window=window).std()
    return df

# Relative Strength Index (RSI): a momentum oscillator that measures the speed and change of price movements. It is used to identify overbought or oversold conditions in a market, typically on a scale from 0 to 100.
# RSI values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions.
def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    
    """Args : df: Date‑indexed DataFrame with 'Close'. window: Window size for RSI (default 14).
    output  : df with 'rsi' column. """
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use simple moving average of gains/losses
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# Volume-based features: analyze trading volume to identify trends and potential reversals in stock prices. Volume can indicate the strength of a price move, with higher volume suggesting stronger conviction.
def compute_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    
    """Compute volume-based features:
    - Rolling average volume & Volume / rolling avg volume
    Args    : df: Date‑indexed DataFrame with 'Volume'. window: Window size for rolling average.
    output  : df with 'vol_ma', 'vol_ratio'. """
    
    df[f"vol_ma_{window}"] = df['Volume'].rolling(window=window).mean()
    df[f"vol_ratio_{window}"] = df['Volume'] / df[f"vol_ma_{window}"]
    return df

def compute_date_features(df: pd.DataFrame) -> pd.DataFrame:
    
    """ Extract calendar features from the Date index.
    Args    : df: Date‑indexed DataFrame.
    output  : df with 'day_of_week', 'month', 'quarter'. """
    
    df['day_of_week'] = df.index.dayofweek    # Monday=0 … Friday=4
    df['month']       = df.index.month
    df['quarter']     = df.index.quarter
    return df

# Orchestrate all feature computations in a single function.
# This function will apply all the feature engineering steps to the cleaned DataFrame.
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all feature computations.
    Args    : df: Cleaned, Date‑indexed DataFrame.
    output : DataFrame with original columns plus new features. """
    
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
    #Loading cleaned data
    df = pd.read_csv(args.input_path, index_col='Date', parse_dates=True)
    print(f"[build_features] Loaded {len(df)} rows")

    #Building features
    df_feat = build_features(df)
    print(f"[build_features] After feature engineering: {len(df_feat)} rows, {df_feat.shape[1]} columns")

    #Saving to CSV
    df_feat.to_csv(args.output_path)
    print(f"[build_features] Features written to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute TSLA features")
    parser.add_argument("--input_path",  type=str, required=True, help="Path to cleaned CSV")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save features CSV")
    args = parser.parse_args()
    main(args)


#head -n 5 data/processed/tsla_features.csv