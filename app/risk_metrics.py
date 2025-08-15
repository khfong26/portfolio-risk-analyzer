import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime

def is_trading_period(start: datetime, end: datetime) -> bool:
    """
    Returns True if the period includes at least one weekday (Mon-Fri).
    """
    days = pd.date_range(start, end)
    return any(day.weekday() < 5 for day in days)

def load_valid_tickers(filename="valid_sp500_tickers.txt"):
    """
    Load valid tickers from a file, return as a list.
    """
    with open(filename, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers

def get_price_data(tickers, start_date, end_date):
    """
    Download price data from Yahoo Finance for the given tickers and date range.
    Returns the adjusted close prices as a DataFrame.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    adj_close = data['Adj Close'] if 'Adj Close' in data else data
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()
    adj_close = adj_close.dropna(axis=0, how='all')
    return adj_close

def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns from price data.
    NaNs are not forward-filled; rows with NaN are dropped.
    """
    return price_df.pct_change(fill_method=None).dropna()

def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    """
    Calculate portfolio returns given asset returns and weights.
    """
    if len(weights) != returns.shape[1]:
        raise ValueError("Weights length must match number of assets")
    if returns.empty:
        return pd.Series(dtype=float)
    weights = np.array(weights)
    return returns.dot(weights)

def portfolio_volatility(portfolio_ret):
    """
    Calculate annualized volatility of portfolio returns.
    """
    return portfolio_ret.std() * np.sqrt(len(portfolio_ret))

def historical_var(portfolio_ret, confidence_level=0.95):
    """
    Calculate the historical Value at Risk (VaR) for the portfolio.
    """
    return np.percentile(portfolio_ret, 100 * (1 - confidence_level))

def parametric_var(portfolio_returns, confidence_level=0.95):
    """
    Calculate the parametric Value at Risk (VaR) for the portfolio.
    Assumes returns are normally distributed.
    """
    mean_return = np.mean(portfolio_returns)
    std_dev = np.std(portfolio_returns)
    z_score = norm.ppf(1 - confidence_level)
    return mean_return + z_score * std_dev

def monte_carlo_var(portfolio_returns, confidence_level=0.95, num_simulations=10000):
    """
    Calculate the Monte Carlo Value at Risk (VaR) for the portfolio.
    Simulates returns using a normal distribution.
    """
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
    return np.percentile(simulated_returns, (1 - confidence_level) * 100)

def run_portfolio_analysis_web(tickers, weights, start_date, end_date, return_prices=False):
    """
    Main analysis function for the web app.
    Returns a dictionary of metrics and (optionally) price/return data.
    """
    prices = get_price_data(tickers, start_date, end_date)
    if prices.empty:
        raise ValueError("No price data found for the given date range.")

    returns = compute_daily_returns(prices)
    port_ret = portfolio_returns(returns, weights)

    if port_ret.empty:
        raise ValueError("No portfolio returns computed. Possibly insufficient data.")

    result = {
        "volatility": portfolio_volatility(port_ret),
        "historical_var": historical_var(port_ret),
        "parametric_var": parametric_var(port_ret),
        "monte_carlo_var": monte_carlo_var(port_ret),
    }

    if return_prices:
        result["prices"] = prices
        result["returns"] = port_ret

    return result