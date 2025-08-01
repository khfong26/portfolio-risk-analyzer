import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from datetime import datetime

def is_trading_period(start, end):
    try:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start, end_date=end)
        return not schedule.empty
    except Exception as e:
        print(f"Error checking trading calendar: {e}")
        return False


def load_valid_tickers(filename="valid_sp500_tickers.txt"):
    with open(filename, "r") as f:
        return set(line.strip().upper() for line in f if line.strip())

def get_user_input():
    valid_tickers = load_valid_tickers()

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

def plot_returns_with_var(portfolio_ret, hist_var, param_var, monte_carlo_var):
    plt.figure(figsize=(10,6))
    plt.hist(portfolio_ret, bins=50, alpha=0.6, color='blue', label='Daily Returns')

    # Add vertical lines for VaRs
    plt.axvline(hist_var, color='red', linestyle='--', label='Historical VaR (95%)')
    plt.axvline(param_var, color='orange', linestyle='--', label='Parametric VaR (95%)')
    plt.axvline(monte_carlo_var, color='green', linestyle='--', label='Monte Carlo VaR (95%)')


    plt.title('Portfolio Daily Returns with VaR Levels')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def monte_carlo_var(portfolio_returns, confidence_level=0.95, num_simulations=10000):
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()

    simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
    var_percentile = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    
    return var_percentile

def get_user_input():
    valid_tickers = load_valid_tickers()

    while True:
        try:
            tickers = input("Enter ticker symbols (comma-separated): ").upper().split(",")
            tickers = [t.strip() for t in tickers if t.strip()]
            if not tickers:
                raise ValueError("Please enter at least one ticker.")

            invalid = [t for t in tickers if t not in valid_tickers]
            if invalid:
                raise ValueError(f"Invalid tickers: {', '.join(invalid)}")

            break
        except Exception as e:
            print(f"Error: {e}. Try again.\n")

    while True:
        try:
            weights_input = input(f"Enter {len(tickers)} weights (comma-separated, must sum to 1): ")
            weights = [float(w.strip()) for w in weights_input.split(",")]
            if len(weights) != len(tickers):
                raise ValueError(f"You entered {len(weights)} weights, expected {len(tickers)}.")
            if not abs(sum(weights) - 1.0) < 1e-5:
                raise ValueError(f"Weights must sum to 1. Current sum: {sum(weights):.4f}")
            break
        except Exception as e:
            print(f"Error: {e}. Try again.\n")

    while True:
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")

        # Check date format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD.")
            continue

        if is_trading_period(start_date, end_date):
            break
        else:
            print("⚠️ No trading days in this date range. Please choose different dates.")


    return tickers, weights, start_date, end_date


def run_portfolio_analysis():
    try:
        tickers, weights, start_date, end_date = get_user_input()
        prices = get_price_data(tickers, start_date, end_date)

        if prices.empty:
            raise ValueError("No price data found for the given date range.")

        returns = compute_daily_returns(prices)
        port_ret = portfolio_returns(returns, weights)

        if port_ret.empty:
            raise ValueError("No portfolio returns computed. Possibly insufficient data.")

        vol = portfolio_volatility(port_ret)
        var_95 = historical_var(port_ret)
        param_var_95 = parametric_var(port_ret)
        mc_var_95 = monte_carlo_var(port_ret)

        print(f"Annualized Volatility: {vol:.2%}")
        print(f"Historical VaR (95%): {var_95:.2%}")
        print(f"Parametric VaR (95%): {param_var_95:.2%}")
        print(f"Monte Carlo VaR (95%): {mc_var_95:.2%}")

        plot_returns_with_var(port_ret, var_95, param_var_95, mc_var_95)

    except ValueError as ve:
        print(f"❌ Input error: {ve}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")



if __name__ == "__main__":
    '''
    #Quick test
    tickers = ['AAPL', 'MSFT', 'TSLA']
    weights = [0.4, 0.4, 0.2]
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    '''
    
    
    run_portfolio_analysis()
