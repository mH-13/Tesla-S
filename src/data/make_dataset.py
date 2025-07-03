"""
make_dataset.py

This script ingests raw Tesla stock CSV data, applies basic cleaning
and preprocessing steps, and writes out a cleaned dataset for modeling.

Usage:
    python -m src.data.make_dataset \
        --input_path ../../data/raw/TSLA.csv \
        --output_path ../../data/processed/tsla_cleaned.csv
"""

import argparse
import pandas as pd


def load_raw_data(input_path: str) -> pd.DataFrame:
    """
    Load raw CSV into a DataFrame.

    Args:
        input_path (str): Path to the raw CSV.
    Returns:
        pd.DataFrame: Raw data.
    """
    df = pd.read_csv(input_path)
    print(f"[load_raw_data] Loaded {len(df)} rows from {input_path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw DataFrame.

    Steps:
    1. Convert 'Date' column to datetime.
    2. Sort rows by date ascending.
    3. Drop duplicates.
    4. Handle missing values:
      - If any OHLC or Volume is missing, drop that row.
    5. Reset index to Date.

    Args:
        df (pd.DataFrame): Raw data.
    Returns:
        pd.DataFrame: Cleaned data, indexed by datetime.
    """
    # 1) Date → datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # 2) Drop rows where Date conversion failed
    bad_dates = df['Date'].isna().sum()
    if bad_dates:
        print(f"[clean_data] Dropping {bad_dates} rows with invalid dates")
    df = df.dropna(subset=['Date'])

    # 3) Sort by Date
    df = df.sort_values('Date')

    # 4) Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['Date'])
    dupes = before - len(df)
    if dupes:
        print(f"[clean_data] Dropped {dupes} duplicate rows based on Date")

    # 5) Drop any rows with missing OHLC or Volume
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_counts = df[required_cols].isna().sum()
    if missing_counts.any():
        print(f"[clean_data] Missing values before drop:\n{missing_counts}")
    df = df.dropna(subset=required_cols)

    # 6) Set Date as index
    df = df.set_index('Date')

    print(f"[clean_data] Final clean dataset has {len(df)} rows")
    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Write the cleaned DataFrame to a CSV.

    Args:
        df (pd.DataFrame): Cleaned data.
        output_path (str): Where to write the processed CSV.
    """
    df.to_csv(output_path)
    print(f"[save_processed_data] Written cleaned data to {output_path}")


def main(args):
    # Load raw
    raw_df = load_raw_data(args.input_path)

    # Clean
    clean_df = clean_data(raw_df)

    # Save
    save_processed_data(clean_df, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest and clean Tesla stock data"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to raw TSLA CSV"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write cleaned CSV"
    )
    args = parser.parse_args()
    main(args)


"""to run this script, use:
python -m src.data.make_dataset \
    --input_path ../../data/raw/TSLA.csv \
    --output_path ../../data/processed/tsla_cleaned.csv
    
    python -m src.data.make_dataset \
    --input_path data/raw/Tasla_Stock_Updated_V2.csv \
    --output_path data/processed/tsla_cleaned.csv

    head -n 5 data/processed/tsla_cleaned.csv
"""
# Note: Adjust paths as necessary based on your project structure.
# This script assumes the raw data is in a specific format with columns:
# 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.