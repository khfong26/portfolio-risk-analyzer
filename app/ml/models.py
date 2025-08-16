import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle

# Optional: import XGBoost and LightGBM if installed
try:
    from xgboost import XGBRegressor
    xgb_installed = True
except ImportError:
    xgb_installed = False

try:
    from lightgbm import LGBMRegressor
    lgbm_installed = True
except ImportError:
    lgbm_installed = False

# Load master features and importances
df = pd.read_csv("sp500_master_features.csv")

# You may want to load/import your feature importances from a file or rerun the importance calculation here.
# For this example, let's assume you have a list of features sorted by importance:
feature_importance_order = [
    'volatility_10d', 'return_21d', 'return_5d', 'return_1d', 'rsi_14', 'volume_avg_21d',
    'macd_hist', 'macd_signal', 'volume_avg_10d', 'momentum_10d', 'macd', 'momentum_21d',
    'ma_50', 'ma_20', 'ma_5'
]

target = 'target_volatility_10d'
ticker_col = 'ticker'

# Experiment settings
N_values = [5, 10, 15]  # Try top 5, 10, 15 features
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "LinearRegression": LinearRegression()
}
if xgb_installed:
    models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
if lgbm_installed:
    models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)

results = []

for N in N_values:
    top_features = feature_importance_order[:N]
    # Prepare data
    df_exp = df.dropna(subset=top_features + [target, ticker_col]).copy()
    # Label encode ticker for tree models, one-hot for linear
    le = LabelEncoder()
    df_exp['ticker_encoded'] = le.fit_transform(df_exp[ticker_col])
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ticker_ohe = ohe.fit_transform(df_exp[[ticker_col]])
    ticker_ohe_df = pd.DataFrame(ticker_ohe, columns=[f"ticker_{cat}" for cat in ohe.categories_[0]], index=df_exp.index)

    for model_name, model in models.items():
        if model_name == "LinearRegression":
            # Use one-hot encoding for ticker
            X = pd.concat([df_exp[top_features], ticker_ohe_df], axis=1)
        else:
            # Use label encoding for ticker
            X = pd.concat([df_exp[top_features], df_exp[['ticker_encoded']]], axis=1)
        y = df_exp[target]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "N_features": N,
            "model": model_name,
            "RMSE": rmse,
            "R2": r2
        })

        print(f"N={N}, Model={model_name}, RMSE={rmse:.6f}, R2={r2:.4f}")

# Save results table
results_df = pd.DataFrame(results)
results_df.to_csv("model_experiment_results.csv", index=False)
print("\nSummary of all experiments:")
print(results_df)

# Save the best model
best_row = results_df.sort_values(by="RMSE").iloc[0]
best_N = best_row["N_features"]
best_model_name = best_row["model"]

print(f"\nBest model: {best_model_name} with top {best_N} features (RMSE={best_row['RMSE']:.6f}, R2={best_row['R2']:.4f})")

# Retrain best model on full data
top_features = feature_importance_order[:int(best_N)]
df_best = df.dropna(subset=top_features + [target, ticker_col]).copy()
le = LabelEncoder()
df_best['ticker_encoded'] = le.fit_transform(df_best[ticker_col])
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
ticker_ohe = ohe.fit_transform(df_best[[ticker_col]])
ticker_ohe_df = pd.DataFrame(ticker_ohe, columns=[f"ticker_{cat}" for cat in ohe.categories_[0]], index=df_best.index)

if best_model_name == "LinearRegression":
    X_best = pd.concat([df_best[top_features], ticker_ohe_df], axis=1)
    best_model = LinearRegression()
else:
    X_best = pd.concat([df_best[top_features], df_best[['ticker_encoded']]], axis=1)
    if best_model_name == "RandomForest":
        best_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif best_model_name == "XGBoost" and xgb_installed:
        best_model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    elif best_model_name == "LightGBM" and lgbm_installed:
        best_model = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError("Unknown model type for retraining.")

best_model.fit(X_best, df_best[target])

# Save the model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Best model saved as best_model.pkl")
print(f"Best model: {best_model_name} with top {best_N} features (RMSE={best_row['RMSE']:.6f}, R2={best_row['R2']:.4f})")