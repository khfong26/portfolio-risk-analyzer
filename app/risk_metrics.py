import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm

def get_price_data(tickers, start_date, end_date):
    # Download price data from Yahoo Finance of the given tickers and date range
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
    return data.dropna()

def compute_daily_returns(price_df):
    # Compute daily returns from price data.
    return price_df.pct_change().dropna()

def portfolio_returns(daily_returns, weights):
    # Calculate portfolio daily returns given individual daily returns 
    # and percentage of each stock in portfolio.
    weights = np.array(weights)
    return daily_returns.dot(weights)

def portfolio_volatility(portfolio_ret):
    # Calculate annualized volatility of portfolio returns.
    return portfolio_ret.std() * np.sqrt(len(portfolio_ret))

def historical_var(portfolio_ret, confidence_level=0.95):
    # Calculate the historical Value at Risk (VaR) for the portfolio.
    # Calculate the percentile of losses (negative returns)
    var = np.percentile(portfolio_ret, 100 * (1 - confidence_level))
    return var

def parametric_var(portfolio_returns, confidence_level=0.95):
    # Calculate the parametric Value at Risk (VaR) for the portfolio.
    # Assuming returns are normally distributed, we calculate the mean and standard deviation.
    mean_return = np.mean(portfolio_returns)
    std_dev = np.std(portfolio_returns)
    z_score = norm.ppf(1 - confidence_level)
    var = (mean_return + z_score * std_dev)
    return var


if __name__ == "__main__":
    # Quick test
    tickers = ['AAPL', 'MSFT', 'TSLA']
    weights = [0.4, 0.4, 0.2]
    start_date = '2022-01-01'
    end_date = '2023-01-01'

    prices = get_price_data(tickers, start_date, end_date)
    returns = compute_daily_returns(prices)
    port_ret = portfolio_returns(returns, weights)
    vol = portfolio_volatility(port_ret)
    var_95 = historical_var(port_ret)

    print(f"Annualized Volatility: {vol:.2%}")
    print(f"Historical VaR (95% confidence): {var_95:.2%}")

    parametric_var_95 = parametric_var(port_ret)
    print(f"Parametric VaR (95% confidence): {parametric_var_95:.2%}")
