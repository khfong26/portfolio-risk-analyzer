from flask import Flask, render_template, request, jsonify
from datetime import datetime
from risk_metrics import load_valid_tickers, run_portfolio_analysis_web
from ml.pipeline import predict  

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    metrics = {
        "volatility": "N/A",
        "historical_var": "N/A",
        "parametric_var": "N/A",
        "monte_carlo_var": "N/A",
        "sharpe_ratio": "N/A",
        "max_drawdown": "N/A",
        "sortino_ratio": "N/A",
        "beta": "N/A",
    }

    if request.method == "POST":
        # Get form data
        tickers = request.form.getlist("tickers[]")
        weights = request.form.getlist("weights[]")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")

        # Convert tickers to uppercase for validation and processing
        tickers = [t.upper() for t in tickers]

        # Validate tickers and weights
        valid_tickers = set(load_valid_tickers())
        try:
            weights = [float(w) for w in weights]
        except ValueError:
            error = "All weights must be numbers."
            return render_template("index.html", error=error)

        if not tickers or not weights or len(tickers) != len(weights):
            error = "Please provide the same number of tickers and weights."
            return render_template("index.html", error=error)

        if not all(t in valid_tickers for t in tickers):
            error = "One or more tickers are invalid."
            return render_template("index.html", error=error)

        if not abs(sum(weights) - 1.0) < 1e-6:
            error = "Weights must sum to 1."
            return render_template("index.html", error=error)

        # Validate dates
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except Exception:
            error = "Invalid date format."
            return render_template("index.html", error=error)

        # Run analysis
        try:
            result = run_portfolio_analysis_web(
                tickers, weights, start_date, end_date
            )
            for key in metrics:
                value = result.get(key)
                metrics[key] = value if value is not None else "N/A"

        except Exception as e:
            error = str(e)

        # Render results page with metrics and error (if any)
        return render_template(
            "results.html",
            error=error,
            **metrics
        )

    # GET request: show the input form
    return render_template("index.html", error=error)

if __name__ == "__main__":
    app.run(debug=True)