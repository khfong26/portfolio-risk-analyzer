# 📊 Portfolio Risk Analyzer

A Streamlit web app that lets users evaluate the risk of their custom stock portfolios using historical market data.

## 🚀 Features

- ✅ Pulls historical price data using `yfinance`
- ✅ Supports user-defined tickers and weights
- ✅ Calculates key risk metrics:
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
  - Volatility
  - Max Drawdown
  - Correlation matrix
- ✅ Clean, interactive dashboard built with Streamlit

## 📦 Tech Stack

- Python 3
- pandas, numpy, scipy
- yfinance
- matplotlib, seaborn
- Streamlit

## 📈 Example Use

```bash
streamlit run app/main.py
