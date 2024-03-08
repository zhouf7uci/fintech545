#probelm 2
import pandas as pd
import numpy as np
from scipy.stats import norm, t

file_path = 'Week05/problem1.csv'
data = pd.read_csv(file_path)

# Calculate VaR using the normal distribution
lambda_ = 0.97
ewma_variance = data['x'].ewm(alpha=(1-lambda_)).var().iloc[-1]
ewma_std = np.sqrt(ewma_variance)
confidence_level = 0.95
VaR_normal = -norm.ppf(1-confidence_level) * ewma_std

# Calculate ES using the normal distribution
ES_normal = (norm.pdf(norm.ppf(1-confidence_level)) / (1-confidence_level)) * ewma_std

# Calculate VaR using the fitted T-distribution
params = t.fit(data['x'])
df, loc, scale = params
VaR_t = -(loc + scale * t.ppf(1-confidence_level, df))

# Calculate ES for the T-distribution
if df > 1:
    x_alpha = t.ppf(1-confidence_level, df)
    ES_t = (df + x_alpha**2) / (df-1) * scale * t.pdf(x_alpha, df) / (1-confidence_level) + loc
else:
    ES_t = np.inf

# Calculate VaR using Historic Simulation
VaR_historic = data['x'].quantile(1 - confidence_level)

# Calculate ES using Historic Simulation
ES_historic = -data[data['x'] <= VaR_historic]['x'].mean()

# Output the results
print(f"VaR (Normal Distribution with EWMA): {VaR_normal}")
print(f"\nES (Normal Distribution with EWMA): {ES_normal}")
print(f"\nVaR (T-distribution): {VaR_t}")
print(f"\nES (T-distribution): {ES_t}")
print(f"\nVaR (Historic Simulation): {-VaR_historic}")
print(f"\nES (Historic Simulation): {ES_historic}")

