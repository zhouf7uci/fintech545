import pandas as pd
import numpy as np
from scipy.stats import norm, t
import statsmodels.api as sm

def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    if dateColumn not in prices.columns:
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {prices.columns}")
    
    vars = prices.columns.difference([dateColumn])
    
    p = prices[vars].values
    n, m = p.shape
    
    # Calculate returns
    p2 = np.zeros((n-1, m))
    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]
    
    # Apply method
    if method.upper() == "DISCRETE":
        p2 = p2 - 1
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\", \"DISCRETE\")")
    
    # Create output DataFrame
    dates = prices[dateColumn].iloc[1:]
    out = pd.DataFrame(data=p2, columns=vars)
    out.insert(0, dateColumn, dates.values)
    
    return out

# Implementation
file_path = '/Users/heyahe/FINTECH-545/Week04/DailyPrices.csv'
prices_df = pd.read_csv(file_path)
returns = return_calculate(prices_df, method="DISCRETE", dateColumn="Date")
print(returns)

# extract returns of META
meta_returns = returns.loc[:,"META"].copy()

# de-mean returns of META by subtracting the mean of returns of META from each return
meta_mean = meta_returns.mean()
meta_returns -= meta_mean
meta_returns = pd.DataFrame(meta_returns)

# Verify that the mean of the centered series is 0
print("\nMean of META de_mean series:", meta_returns.mean()[0])

# Using a Normal Distribution
confidence_level = 0.95
mean_returns = meta_returns.mean()
std_dev_returns = meta_returns.std()

VaR_normal = -norm.ppf(1 - confidence_level) * std_dev_returns + mean_returns
print("\nVaR using Normal Distribution: {:.4f}%".format(VaR_normal[0]*100))

# Using a Normal Distribution with an Exponentially Weighted Variance
lambda_param = 0.94
ewma_variance = meta_returns.ewm(alpha=(1 - lambda_param)).var().iloc[-1]
ewma_std_dev = ewma_variance ** 0.5

VaR_normal_EWMA = -norm.ppf(1 - confidence_level) * ewma_std_dev + mean_returns
print("VaR using Normal Distribution with EWMA: {:.4f}%".format(VaR_normal_EWMA[0]*100))

# Using a MLE Fitted T Distribution
# Fit a T distribution to META returns
params = t.fit(meta_returns)

# Calculate VaR from the fitted T distribution
VaR_t = -t.ppf(1 - confidence_level, *params) * std_dev_returns + mean_returns
print("VaR using MLE Fitted T Distribution: {:.4f}%".format(VaR_t[0]*100))

# Using a Fitted AR(1) Model
# Fit an AR(1) model
model = sm.tsa.ARIMA(meta_returns, order=(1, 0, 0))
model_fitted = model.fit()

# Predict the next value
prediction = model_fitted.predict(start=len(returns), end=len(returns))

# Calculate the standard deviation of residuals
residuals_std_dev = model_fitted.resid.std()

# Calculate VaR
VaR_AR1 = -norm.ppf(1 - confidence_level) * residuals_std_dev
print("VaR using AR(1) Model: {:.4f}%".format(VaR_AR1*100))

# Using a Historic Simulation
# VaR_historic = -meta_returns.quantile(1 - confidence_level)
VaR_historic = meta_returns.mean() - np.quantile(meta_returns,0.05)
print("VaR using Historic Simulation: {:.4f}%".format(VaR_historic[0]*100))
