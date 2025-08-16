import os
import pandas as pd
import numpy as np

def load_and_combine_csvs(folder):
    """
    Load all CSVs from a folder, add ticker column, and combine into a single DataFrame.
    """
    all_dfs = []
    for fname in os.listdir(folder):
        if fname.endswith(".csv"):
            ticker = fname.replace(".csv", "")
            df = pd.read_csv(os.path.join(folder, fname), parse_dates=["Date"])
            df["ticker"] = ticker
            all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined

def clean_and_sort(df):
    """
    Ensure consistent date formatting, sort by ticker and date, and handle missing values.
    """
    df = df.sort_values(["ticker", "Date"])
    df = df.dropna(subset=["Close"])  # Drop rows where price is missing
    df = df.reset_index(drop=True)
    return df

def compute_rsi(series, window=14):
    """
    Compute the Relative Strength Index (RSI) for a price series.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    """
    Compute MACD and signal line for a price series.
    """
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def engineer_features(df):
    """
    Engineer features per ticker: returns, volatility, moving averages, momentum, RSI, MACD, volume metrics, etc.
    """
    feature_dfs = []
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("Date").copy()
        group["return_1d"] = group["Close"].pct_change()
        group["return_5d"] = group["Close"].pct_change(5)
        group["return_21d"] = group["Close"].pct_change(21)
        group["volatility_10d"] = group["return_1d"].rolling(10).std()
        group["volatility_21d"] = group["return_1d"].rolling(21).std()
        group["ma_5"] = group["Close"].rolling(5).mean()
        group["ma_20"] = group["Close"].rolling(20).mean()
        group["ma_50"] = group["Close"].rolling(50).mean()
        group["momentum_10d"] = group["Close"] - group["Close"].shift(10)
        group["momentum_21d"] = group["Close"] - group["Close"].shift(21)
        group["volume_avg_10d"] = group["Volume"].rolling(10).mean()
        group["volume_avg_21d"] = group["Volume"].rolling(21).mean()
        group["rsi_14"] = compute_rsi(group["Close"], window=14)
        macd, signal = compute_macd(group["Close"])
        group["macd"] = macd
        group["macd_signal"] = signal
        group["macd_hist"] = group["macd"] - group["macd_signal"]
        # Drawdown
        group["cum_return"] = (1 + group["return_1d"]).cumprod()
        group["cum_max"] = group["cum_return"].cummax()
        group["drawdown"] = (group["cum_return"] - group["cum_max"]) / group["cum_max"]
        feature_dfs.append(group)
    features = pd.concat(feature_dfs, ignore_index=True)
    return features

def add_target_variables(df, n_forward=10, vol_thresh=0.03, dd_thresh=-0.1):
    """
    Add multiple target variables for modeling:
    - N-day forward volatility
    - N-day forward max drawdown
    - Binary: volatility exceeds threshold
    - Binary: drawdown exceeds threshold
    """
    target_dfs = []
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("Date").copy()
        # N-day forward volatility
        group["target_volatility_10d"] = group["return_1d"].rolling(n_forward).std().shift(-n_forward)
        # N-day forward max drawdown
        group["target_max_drawdown_10d"] = group["drawdown"].rolling(n_forward).min().shift(-n_forward)
        # Binary: volatility exceeds threshold
        group["target_high_vol"] = (group["target_volatility_10d"] > vol_thresh).astype(int)
        # Binary: drawdown exceeds threshold
        group["target_high_dd"] = (group["target_max_drawdown_10d"] < dd_thresh).astype(int)
        target_dfs.append(group)
    targets = pd.concat(target_dfs, ignore_index=True)
    return targets

def save_master_dataset(df, output_path):
    """
    Save the final DataFrame as CSV or Parquet.
    """
    if output_path.endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # 1. Load and combine
    combined = load_and_combine_csvs("sp500_data")
    # 2. Clean and sort
    cleaned = clean_and_sort(combined)
    # 3. Engineer features
    features = engineer_features(cleaned)
    # 4. Add target variables
    final = add_target_variables(features, n_forward=10, vol_thresh=0.03, dd_thresh=-0.1)
    # 5. Save master dataset
    save_master_dataset(final, "sp500_master_features.csv")