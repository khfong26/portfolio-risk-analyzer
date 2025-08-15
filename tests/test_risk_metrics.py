import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.risk_metrics import (
    compute_daily_returns,
    is_trading_period,
    load_valid_tickers,
    get_price_data,
    portfolio_returns,
)

# --- compute_daily_returns tests ---

def test_compute_daily_returns_basic():
    df = pd.DataFrame({'A': [100, 102, 101]})
    returns = compute_daily_returns(df)
    expected = [0.02, -0.00980392]
    assert np.allclose(returns['A'].values, expected, atol=1e-6)

def test_compute_daily_returns_empty():
    df = pd.DataFrame()
    returns = compute_daily_returns(df)
    assert returns.empty

def test_compute_daily_returns_nan():
    df = pd.DataFrame({'A': [100, np.nan, 102]})
    returns = compute_daily_returns(df)
    assert returns.empty  # No valid returns can be computed after a NaN

# --- is_trading_period tests ---

def test_is_trading_period_true():
    start = datetime(2024, 6, 3)
    end = datetime(2024, 6, 7)
    assert is_trading_period(start, end) is True

def test_is_trading_period_false():
    start = datetime(2024, 6, 8)  # Saturday
    end = datetime(2024, 6, 9)    # Sunday
    assert is_trading_period(start, end) is False

def test_is_trading_period_same_day():
    day = datetime(2024, 6, 5)
    assert is_trading_period(day, day) is True

# --- load_valid_tickers tests ---

def test_load_valid_tickers_default():
    tickers = load_valid_tickers()
    assert isinstance(tickers, list)
    assert "AAPL" in tickers or "GOOG" in tickers  # Should contain common tickers

def test_load_valid_tickers_custom(tmp_path):
    file = tmp_path / "tickers.txt"
    file.write_text("AAA\nBBB\nCCC\n")
    tickers = load_valid_tickers(str(file))
    assert tickers == ["AAA", "BBB", "CCC"]

# --- get_price_data tests ---

def test_get_price_data_valid(monkeypatch):
    # Mock yfinance download
    import app.risk_metrics as rm
    def mock_download(tickers, start, end, **kwargs):
        dates = pd.date_range(start, end)
        data = pd.DataFrame(
            {t: np.linspace(100, 110, len(dates)) for t in tickers},
            index=dates
        )
        return data
    monkeypatch.setattr(rm, "yf", type("yf", (), {"download": staticmethod(mock_download)}))
    tickers = ["AAPL", "MSFT"]
    start = "2024-06-01"
    end = "2024-06-10"
    df = get_price_data(tickers, start, end)
    assert set(df.columns) == set(tickers)
    assert len(df) > 0

def test_get_price_data_empty(monkeypatch):
    import app.risk_metrics as rm
    def mock_download(tickers, start, end, **kwargs):
        return pd.DataFrame()
    monkeypatch.setattr(rm, "yf", type("yf", (), {"download": staticmethod(mock_download)}))
    df = get_price_data(["FAKE"], "2024-06-01", "2024-06-10")
    assert df.empty

# --- portfolio_returns tests ---

def test_portfolio_returns_basic():
    returns = pd.DataFrame({'A': [0.01, 0.02], 'B': [0.03, -0.01]})
    weights = [0.5, 0.5]
    port_ret = portfolio_returns(returns, weights)
    assert np.allclose(port_ret.values, [0.02, 0.005])

def test_portfolio_returns_weight_mismatch():
    returns = pd.DataFrame({'A': [0.01, 0.02], 'B': [0.03, -0.01]})
    weights = [1.0]  # Wrong length
    with pytest.raises(ValueError):
        portfolio_returns(returns, weights)

def test_portfolio_returns_nan():
    returns = pd.DataFrame({'A': [0.01, np.nan], 'B': [0.03, 0.01]})
    weights = [0.5, 0.5]
    port_ret = portfolio_returns(returns, weights)
    assert np.isnan(port_ret.iloc[1])

def test_portfolio_returns_empty():
    returns = pd.DataFrame()
    weights = []
    port_ret = portfolio_returns(returns, weights)
    assert port_ret.empty