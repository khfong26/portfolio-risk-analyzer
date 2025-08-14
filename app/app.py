from flask import Flask, render_template, request
import pandas as pd
import os
from datetime import datetime
from risk_metrics import load_valid_tickers, run_portfolio_analysis_web
from plotting import plot_cumulative_returns, plot_return_histogram, plot_monte_carlo

app = Flask(__name__)

# Load valid tickers once at startup
valid_tickers = load_valid_tickers()

# Ensure static/graphs folder exists
GRAPH_FOLDER = os.path.join(app.root_path, "static", "graphs")
os.makedirs(GRAPH_FOLDER, exist_ok=True)



# Metric explanations
metric_explanations = {
    "volatility": "Annualized standard deviation of portfolio returns.",
    "historical_var": "95% historical Value at Risk: worst expected loss based on historical returns.",
    "parametric_var": "95% parametric Value at Risk: assumes normal distribution of returns.",
    "monte_carlo_var": "95% Monte Carlo Value at Risk: simulated losses using random sampling."
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        tickers = request.form.getlist("tickers[]")
        weights = request.form.getlist("weights[]")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")

        # --- Clean & validate tickers ---
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        invalid = [t for t in tickers if t not in valid_tickers]
        if invalid:
            return render_template("index.html", error=f"Invalid tickers: {', '.join(invalid)}")

        # --- Clean & validate weights ---
        try:
            weights = [float(w) for w in weights if w.strip()]
        except ValueError:
            return render_template("index.html", error="Weights must be numbers.")

        if len(weights) != len(tickers):
            return render_template("index.html", error="Number of weights must match number of tickers.")
        
        if not abs(sum(weights) - 1.0) < 1e-5:
            return render_template("index.html", error="Weights must sum to 1.0")

        # --- Validate date inputs ---
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return render_template("index.html", error="Dates must be in YYYY-MM-DD format.")

        # --- Run portfolio analysis ---
        try:
            result = run_portfolio_analysis_web(tickers, weights, start_date, end_date, return_prices=True)
            metrics = {k: result[k] for k in ["volatility", "historical_var", "parametric_var", "monte_carlo_var"]}
            price_data = result["returns"]
        except ValueError as e:
            return render_template("index.html", error=str(e))
        except Exception as e:
            return render_template("index.html", error=f"Unexpected error: {e}")

        # --- Generate graphs ---
        # filenames
        graph_filenames = {
            "cumulative_returns": "cumulative_returns.png",
            "return_histogram": "return_histogram.png",
            "monte_carlo": "monte_carlo.png"
        }

        graph_paths = {name: os.path.join(GRAPH_FOLDER, filename) for name, filename in graph_filenames.items()}

        plot_cumulative_returns(price_data, save_path=graph_paths["cumulative_returns"])
        plot_return_histogram(price_data, save_path=graph_paths["return_histogram"])
        plot_monte_carlo(price_data, save_path=graph_paths["monte_carlo"])

        graph_urls = {name: "/static/graphs/" + filename for name, filename in graph_filenames.items()}



        # --- Prepare portfolio table ---
        df = pd.DataFrame({"Ticker": tickers, "Weight": weights})

        return render_template(
            "results.html",
            tables=[df.to_html(index=False)],
            titles=df.columns.values,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            graphs=graph_urls,
            metric_explanations=metric_explanations
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
