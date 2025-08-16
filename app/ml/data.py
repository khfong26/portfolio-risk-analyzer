import yfinance as yf
import pandas as pd
import os

def download_data(ticker, start_date, end_date):
    """
    Download historical price data for a given ticker from Yahoo Finance.
    Returns a DataFrame with date as index.
    """
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    return df

def download_multiple(tickers, start_date, end_date):
    """
    Download historical price data for multiple tickers.
    Returns a dictionary of DataFrames keyed by ticker.
    Skips tickers with no data and prints a warning.
    """
    data = {}
    for ticker in tickers:
        df = download_data(ticker, start_date, end_date)
        if not df.empty:
            data[ticker] = df
        else:
            print(f"Warning: No data found for {ticker}")
    return data

def save_data_to_csv(data_dict, output_folder):
    """
    Save each ticker's DataFrame to a separate CSV file in the output folder.
    The index (date) is saved as a column named 'Date'.
    After saving, remove any unwanted second line that starts with a comma.
    """
    os.makedirs(output_folder, exist_ok=True)
    for ticker, df in data_dict.items():
        df = df.copy()
        df.index.name = "Date"
        df.reset_index(inplace=True)
        csv_path = os.path.join(output_folder, f"{ticker}.csv")
        df.to_csv(csv_path, index=False)
        # Post-process: remove bad second line if present
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > 1 and lines[1].startswith(","):
            print(f"Fixing {ticker}.csv: removing bad second line.")
            lines = [lines[0]] + lines[2:]
            with open(csv_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        print(f"Saved {ticker} to {csv_path}")

if __name__ == "__main__":
    # Read tickers from valid_sp500_tickers.txt
    tickers_file = "valid_sp500_tickers.txt"  # Adjust path if needed
    with open(tickers_file) as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Download data for all tickers
    start_date = "2010-01-01"
    end_date = "2024-01-01"
    data = download_multiple(tickers, start_date, end_date)

    # Save each ticker's data as a CSV in the 'sp500_data' folder
    save_data_to_csv(data, "sp500_data")