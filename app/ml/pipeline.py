import yfinance as yf
import pandas as pd

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
    """
    data = {}
    for ticker in tickers:
        df = download_data(ticker, start_date, end_date)
        data[ticker] = df
    return data

if __name__ == "__main__":
    # Example usage
    df = download_data("AAPL", "2020-01-01", "2024-01