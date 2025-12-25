# ==============================
# STEP 1: DOWNLOAD MARKET DATA
# ==============================

import yfinance as yf
import pandas as pd

TICKER = "SPY"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

# Download data (auto-adjusted prices)
prices = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True
)

# Keep only Close price (already adjusted)
prices = prices[['Close']]
prices.rename(columns={'Close': 'Price'}, inplace=True)

prices.dropna(inplace=True)

# Save data
prices.to_csv("prices.csv")

print("STEP 1 COMPLETE: Data downloaded successfully")
print(prices.head())
# ==============================
# STEP 2: CALCULATE DAILY RETURNS
# ==============================

# Reload prices
prices = pd.read_csv("prices.csv", index_col=0)

# If columns are multi-level, flatten them
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.get_level_values(-1)

# Select first column (price column)
price_col = prices.columns[0]

# Convert prices to numeric (CRITICAL FIX)
prices[price_col] = pd.to_numeric(prices[price_col], errors="coerce")

# Drop any non-numeric rows
prices.dropna(inplace=True)

# Calculate daily returns (disable deprecated fill)
returns = prices[price_col].pct_change(fill_method=None).dropna()

# Save returns
returns.to_csv("returns.csv", header=["Return"])

print("\nSTEP 2 COMPLETE: Daily returns calculated")
print(returns.describe())
# ==============================
# STEP 3: HISTORICAL VALUE AT RISK
# ==============================

import numpy as np

# Load returns
returns = pd.read_csv("returns.csv")

# Convert to numeric (safety)
returns['Return'] = pd.to_numeric(returns['Return'], errors='coerce')
returns.dropna(inplace=True)

# Confidence level
CONFIDENCE_LEVEL = 0.95

# Historical VaR calculation
historical_var = np.percentile(
    returns['Return'],
    (1 - CONFIDENCE_LEVEL) * 100
)

print("\nSTEP 3 COMPLETE: Historical VaR calculated")
print(f"Historical VaR (95%): {historical_var:.4f}")

# ==============================
# STEP 4: PARAMETRIC (NORMAL) VAR
# ==============================

from scipy.stats import norm

# Load returns
returns = pd.read_csv("returns.csv")
returns['Return'] = pd.to_numeric(returns['Return'], errors='coerce')
returns.dropna(inplace=True)

# Calculate mean and standard deviation
mu = returns['Return'].mean()
sigma = returns['Return'].std()

# Parametric VaR calculation
parametric_var = norm.ppf(
    1 - CONFIDENCE_LEVEL,
    mu,
    sigma
)

print("\nSTEP 4 COMPLETE: Parametric VaR calculated")
print(f"Mean return: {mu:.5f}")
print(f"Volatility (std): {sigma:.5f}")
print(f"Parametric VaR (95%): {parametric_var:.4f}")

# ==============================
# STEP 5: MONTE CARLO VAR
# ==============================

import numpy as np

# Number of simulations
NUM_SIMULATIONS = 10000

# Generate simulated returns
simulated_returns = np.random.normal(
    loc=mu,
    scale=sigma,
    size=NUM_SIMULATIONS
)

# Monte Carlo VaR calculation
monte_carlo_var = np.percentile(
    simulated_returns,
    (1 - CONFIDENCE_LEVEL) * 100
)

print("\nSTEP 5 COMPLETE: Monte Carlo VaR calculated")
print(f"Monte Carlo VaR (95%): {monte_carlo_var:.4f}")
# ==============================
# STEP 6: SUMMARY & OUTPUT
# ==============================

# Create summary table
var_summary = pd.DataFrame({
    "Method": [
        "Historical VaR",
        "Parametric VaR",
        "Monte Carlo VaR"
    ],
    "VaR (95%)": [
        historical_var,
        parametric_var,
        monte_carlo_var
    ]
})

# Save summary
var_summary.to_csv("var_summary.csv", index=False)

print("\nSTEP 6 COMPLETE: VaR Summary")
print(var_summary)
# ==============================
# STEP 7: VISUALIZATION (HISTOGRAM)
# ==============================

import matplotlib.pyplot as plt

# Load returns again (safe practice)
returns = pd.read_csv("returns.csv")
returns['Return'] = pd.to_numeric(returns['Return'], errors='coerce')
returns.dropna(inplace=True)

plt.figure(figsize=(10, 6))

# Histogram of returns
plt.hist(returns['Return'], bins=50, alpha=0.75, edgecolor='black')

# VaR lines
plt.axvline(historical_var, color='red', linestyle='--', linewidth=2,
            label='Historical VaR (95%)')

plt.axvline(parametric_var, color='green', linestyle='--', linewidth=2,
            label='Parametric VaR (95%)')

plt.axvline(monte_carlo_var, color='purple', linestyle='--', linewidth=2,
            label='Monte Carlo VaR (95%)')

plt.title("Return Distribution with Value at Risk (VaR)")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("var_histogram.png", dpi=300)
plt.show()
