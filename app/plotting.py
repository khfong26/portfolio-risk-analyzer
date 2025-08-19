import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  


def plot_cumulative_returns(portfolio_df, save_path=None):
    plt.figure(figsize=(10,6))
    (1 + portfolio_df).cumprod().plot()
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_return_histogram(portfolio_df, save_path=None):
    plt.figure(figsize=(10,6))
    portfolio_df.hist(bins=50)
    plt.title("Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_monte_carlo(portfolio_df, save_path=None):
    plt.figure(figsize=(10,6))
    for _ in range(100):
        sim = (1 + portfolio_df.sample(frac=1, replace=True)).cumprod()
        plt.plot(sim, alpha=0.1, color='blue')
    plt.title("Monte Carlo Simulation")
    if save_path:
        plt.savefig(save_path)
    plt.close()

