# Portfolio Risk Analyzer

A Flask-based web application for analyzing stock portfolios, calculating risk metrics, and visualizing returns. This tool helps users understand portfolio volatility, Value at Risk (VaR), and cumulative performance through interactive graphs.

---

## Features

- **Portfolio Metrics**
  - Annualized volatility
  - Historical VaR (95%)
  - Parametric VaR (95%)
  - Monte Carlo VaR (95%)

- **Graphs**
  - Cumulative returns over time
  - Return distribution histogram
  - Monte Carlo simulation of portfolio performance

- **Validations**
  - Only valid S&P 500 tickers accepted
  - Weights must sum to 1
  - Date range must include trading days

- **Web Interface**
  - User-friendly form for entering tickers, weights, and dates
  - Results displayed in tables and interactive graphs

---

## Installation

1. Clone the repository:  
   git clone https://github.com/khfong26/portfolio-risk-analyzer.git
   cd portfolio-risk-analyzer/app
2. Create and activate a virtual environment:
  python -m venv venv
  source venv/bin/activate      # Linux/Mac
  venv\Scripts\activate         # Windows
3. Install dependencies:
  pip install -r requirements.txt
4. Run the app:
  python app.py

## Usage
Enter ticker symbols from the S&P 500.
Enter corresponding portfolio weights (must sum to 1).
Select a valid start and end date (trading days only).
Submit the form to view portfolio metrics and graphs.

