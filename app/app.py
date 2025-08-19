from flask import Flask, render_template, request
from datetime import datetime
from risk_metrics import load_valid_tickers, run_portfolio_analysis_web
from ml.pipeline import predict_portfolio as predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None
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
        mode = request.form.get("mode")
        tickers = []
        weights = None

        if mode == "ml":
            # ML mode: get comma-separated tickers and weights
            tickers_str = request.form.get("tickers[]", "")
            tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            weights_str = request.form.get("weights[]", "")
            weights = [float(w.strip()) for w in weights_str.split(",") if w.strip()] if weights_str else None
        elif mode == "var":
            # VaR mode: get tickers and weights from repeated fields
            tickers = request.form.getlist("tickers[]")
            tickers = [t.strip().upper() for t in tickers if t.strip()]
            weights = request.form.getlist("weights[]")
            try:
                weights = [float(w) for w in weights]
            except ValueError:
                error = "All weights must be numbers."
                return render_template("index.html", error=error)

        # Validate tickers
        valid_tickers = set(load_valid_tickers())
        if not tickers or not all(t in valid_tickers for t in tickers):
            error = "Please provide valid S&P 500 tickers."
            return render_template("index.html", error=error)

        if mode == "var":
            # VaR calculation mode
            start_date = request.form.get("start_date")
            end_date = request.form.get("end_date")

            if not weights or len(tickers) != len(weights):
                error = "Please provide the same number of tickers and weights."
                return render_template("index.html", error=error)

            if not abs(sum(weights) - 1.0) < 1e-6:
                error = "Weights must sum to 1."
                return render_template("index.html", error=error)

            try:
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except Exception:
                error = "Invalid date format."
                return render_template("index.html", error=error)

            try:
                result = run_portfolio_analysis_web(
                    tickers, weights, start_date, end_date
                )
                for key in metrics:
                    value = result.get(key)
                    metrics[key] = value if value is not None else "N/A"
            except Exception as e:
                error = str(e)

            return render_template(
                "results.html",
                error=error,
                mode="var",
                **metrics
            )

        elif mode == "ml":
            # ML prediction mode (multiple tickers)
            try:
                results, weighted_avg = predict(tickers, weights if weights and len(weights) == len(tickers) else None)
            except Exception as e:
                error = f"Prediction error: {e}"
                return render_template("index.html", error=error)

            return render_template(
                "results.html",
                error=error,
                mode="ml",
                ml_results=results,
                ml_weighted_avg=weighted_avg
            )

    # GET request: show the input form
    return render_template("index.html", error=error)

if __name__ == "__main__":
    app.run(debug=True)