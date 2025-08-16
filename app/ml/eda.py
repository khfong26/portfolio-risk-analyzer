import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the master dataset
df = pd.read_csv("sp500_master_features.csv")

# Basic info and summary
print("=== DataFrame Info ===")
print(df.info())
print("\n=== DataFrame Describe ===")
print(df.describe(include='all'))
print("\n=== Head ===")
print(df.head())

# Check for missing values
print("\n=== Missing Values (per column) ===")
print(df.isnull().sum())

# Optionally, drop rows with missing target or features
# df = df.dropna(subset=['target_volatility_10d', 'return_1d', ...])

# Fill missing values for features (example: forward fill, then back fill)
df = df.fillna(method='ffill').fillna(method='bfill')

# Plot distributions for key features
features_to_plot = [
    'return_1d', 'volatility_10d', 'ma_5', 'momentum_10d', 'rsi_14', 'macd', 'volume_avg_10d'
]
for feature in features_to_plot:
    if feature in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[feature].dropna(), bins=100, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

# Plot distribution for target variable(s)
targets_to_plot = [
    'target_volatility_10d', 'target_max_drawdown_10d', 'target_high_vol', 'target_high_dd'
]
for target in targets_to_plot:
    if target in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[target].dropna(), bins=100, kde=True)
        plt.title(f'Distribution of {target}')
        plt.show()

# Correlation heatmap for numeric features
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Example: time series plot for a single ticker
sample_ticker = df['ticker'].iloc[0]
df_sample = df[df['ticker'] == sample_ticker]
plt.figure(figsize=(12, 4))
plt.plot(df_sample['Date'], df_sample['Close'])
plt.title(f'Close Price Over Time: {sample_ticker}')
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()