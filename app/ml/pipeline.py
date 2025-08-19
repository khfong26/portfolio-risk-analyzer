import joblib
import pandas as pd

# Load model and label encoder
model = joblib.load("xgb_model_13_features.pkl")
le = joblib.load("label_encoder_xgb.pkl")

# Must match training!
top_features = [
    'return_21d', 'return_5d', 'return_1d', 'rsi_14', 'volume_avg_21d',
    'macd_hist', 'macd_signal', 'volume_avg_10d', 'momentum_10d', 'macd',
    'ma_50', 'momentum_21d', 'ma_20'
]

def predict(input_dict):
    """
    input_dict: {"return_21d": val, ..., "ma_20": val, "ticker": "AAPL"}
    Returns: float prediction (10-day volatility)
    """
    df = pd.DataFrame([input_dict])
    # Encode ticker
    df["ticker_encoded"] = le.transform(df["ticker"])
    # Select features
    X = pd.concat([df[top_features], df[["ticker_encoded"]]], axis=1)
    # Predict
    return float(model.predict(X)[0])