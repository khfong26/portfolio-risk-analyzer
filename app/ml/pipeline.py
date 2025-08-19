import joblib
import pandas as pd
import numpy as np
import os
import yfinance as yf

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "xgb_model_13_features.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_xgb.pkl")

model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

top_features = [
    'return_21d', 'return_5d', 'return_1d', 'rsi_14', 'volume_avg_21d',
    'macd_hist', 'macd_signal', 'volume_avg_10d', 'momentum_10d', 'macd',
    'ma_50', 'momentum_21d', 'ma_20'
]

def compute_features(df):
    df = df.copy()
    df = df.sort_index()  # Ensure date ascending
    if len(df) < 60:
        raise ValueError("Not enough data to compute all features (need at least 60 days).")
    df['return_21d'] = df['Close'].pct_change(21)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_1d'] = df['Close'].pct_change(1)
    df['rsi_14'] = compute_rsi(df['Close'], 14)
    df['volume_avg_21d'] = df['Volume'].rolling(21).mean()
    df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(df['Close'])
    df['volume_avg_10d'] = df['Volume'].rolling(10).mean()
    df['momentum_10d'] = df['Close'] - df['Close'].shift(10)
    df['ma_50'] = df['Close'].rolling(50).mean()
    df['momentum_21d'] = df['Close'] - df['Close'].shift(21)
    df['ma_20'] = df['Close'].rolling(20).mean()
    last_row = df.iloc[-1]
    missing = []
    for f in top_features:
        # Check if column exists
        if f not in df.columns:
            missing.append(f)
            continue
        value = last_row[f]
        # If value is a Series (shouldn't be), take the last value
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        # Now check for NaN
        try:
            if pd.isnull(value) or (isinstance(value, float) and np.isnan(value)):
                missing.append(f)
        except Exception:
            missing.append(f)
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)} (not enough data?)")
    return {k: last_row[k] for k in top_features}

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist

def predict_portfolio(tickers, weights=None):
    results = []
    for ticker in tickers:
        try:
            # Fetch more data to ensure all rolling windows are valid
            data = yf.download(ticker, period="1y", progress=False)
            if data.empty or len(data) < 60:
                raise ValueError("Not enough data for " + ticker)
            feats = compute_features(data)
            feats['ticker'] = ticker
            feats['ticker_encoded'] = le.transform([ticker])[0]
            # Ensure all features are float
            for k in top_features:
                feats[k] = float(feats[k])
            X = pd.DataFrame([{**{k: feats[k] for k in top_features}, "ticker_encoded": feats["ticker_encoded"]}])
            X = X.astype(float)
            pred = float(model.predict(X)[0])
            results.append({"ticker": ticker, "prediction": pred, "error": None})
        except Exception as e:
            results.append({"ticker": ticker, "prediction": None, "error": str(e)})
    # Weighted average (if weights provided and all predictions are valid)
    weighted_avg = None
    if weights and all(r.get("prediction") is not None for r in results):
        preds = np.array([r["prediction"] for r in results])
        weights = np.array(weights)
        weighted_avg = float(np.dot(preds, weights))
    return results, weighted_avg