import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

# 1. Load the master dataset
df = pd.read_csv("sp500_master_features.csv")

# 2. Choose candidate features and target
features = [
    'return_1d', 'volatility_10d', 'ma_5', 'momentum_10d', 'rsi_14', 'macd', 'volume_avg_10d',
    'return_5d', 'return_21d', 'ma_20', 'ma_50', 'momentum_21d', 'volume_avg_21d', 'macd_signal', 'macd_hist'
]
target = 'target_volatility_10d'

# 3. Drop rows with missing values in selected features/target
df_model = df.dropna(subset=features + [target])

# 4. Prepare X and y for both cases
# --- Case 1: Without ticker ---
X1 = df_model[features]
y = df_model[target]

# --- Case 2: With ticker (label encoded) ---
le = LabelEncoder()
df_model['ticker_encoded'] = le.fit_transform(df_model['ticker'])
X2 = df_model[features + ['ticker_encoded']]

# 5. Train/test split (same split for fair comparison)
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
X2_train, X2_test, _, _ = train_test_split(X2, y, test_size=0.2, random_state=42)

# 6. Fit Random Forests
rf1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf1.fit(X1_train, y_train)
rf2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf2.fit(X2_train, y_train)

# 7. Evaluate performance
y_pred1 = rf1.predict(X1_test)
y_pred2 = rf2.predict(X2_test)

print("=== Without ticker ===")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred1):.6f}")
print(f"R2: {r2_score(y_test, y_pred1):.4f}")

print("\n=== With ticker (encoded) ===")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred2):.6f}")
print(f"R2: {r2_score(y_test, y_pred2):.4f}")

# 8. Feature importances
importances1 = pd.Series(rf1.feature_importances_, index=features).sort_values(ascending=False)
importances2 = pd.Series(rf2.feature_importances_, index=features + ['ticker_encoded']).sort_values(ascending=False)

print("\nFeature importances (without ticker):")
print(importances1)
print("\nFeature importances (with ticker):")
print(importances2)

# 9. Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances1.values, y=importances1.index)
plt.title("Feature Importances WITHOUT Ticker")
plt.show()
plt.savefig("feature_importances_without_ticker.png")  # Save the plot

plt.figure(figsize=(10, 6))
sns.barplot(x=importances2.values, y=importances2.index)
plt.title("Feature Importances WITH Ticker")
plt.show()
plt.savefig("feature_importances_with_ticker.png")  # Save the plot