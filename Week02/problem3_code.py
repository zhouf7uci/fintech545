import pandas as pd
import numpy as np
from scipy.stats import norm

# Load the portfolio data
portfolio_df = pd.read_csv('/Users/heyahe/FINTECH-545/Week04/portfolio.csv')

# Load the daily prices data
daily_prices_df = pd.read_csv('/Users/heyahe/FINTECH-545/Week04/DailyPrices.csv')

# Copy codes from problem 2
def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    if dateColumn not in prices.columns:
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {prices.columns}")
    vars = prices.columns.difference([dateColumn])
    
    p = prices[vars].values
    n, m = p.shape
    
    p2 = np.zeros((n-1, m))
    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]
    
    if method.upper() == "DISCRETE":
        p2 = p2 - 1
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\", \"DISCRETE\")")

    dates = prices[dateColumn].iloc[1:]
    out = pd.DataFrame(data=p2, columns=vars)
    out.insert(0, dateColumn, dates.values)
    return out

# Calculate daily returns
daily_returns_df = return_calculate(daily_prices_df, method="DISCRETE")

# Set the Date column as the index
daily_returns_df.set_index('Date', inplace=True)

# Calculate Exponentially Weighted Covariance Matrix
lambda_factor = 0.94
ew_cov_matrix = daily_returns_df.ewm(span=(2/(1-lambda_factor))-1, adjust=False).cov(pairwise=True)
latest_covariances = ew_cov_matrix.groupby(level=1).tail(1).reset_index(level=0, drop=True)

# Initialize a dictionary to hold portfolio variances
portfolio_variances = {}
for portfolio_name, portfolio_data in portfolio_df.groupby('Portfolio'):
    portfolio_variance = 0
    for i, row_i in portfolio_data.iterrows():
        for j, row_j in portfolio_data.iterrows():
            weight_i = row_i['Holding']
            weight_j = row_j['Holding']
            try:
                cov_ij = latest_covariances.loc[(row_i['Stock'], row_j['Stock'])]
            except KeyError:
                cov_ij = 0
            portfolio_variance += weight_i * weight_j * cov_ij
    portfolio_variances[portfolio_name] = portfolio_variance

# Define the confidence level, e.g., 95%
confidence_level = 0.95
z_score = norm.ppf(confidence_level)

# Calculate VaR for each portfolio
portfolio_VaRs = {portfolio: z_score * (variance ** 0.5) for portfolio, variance in portfolio_variances.items()}

# Calculate the total holdings by summing up holdings across all portfolios
total_holdings = portfolio_df.groupby('Stock')['Holding'].sum()

# Calculate the total portfolio variance
total_portfolio_variance = 0
for i, stock_i in total_holdings.items():
    for j, stock_j in total_holdings.items():
        cov_ij = ew_cov_matrix.loc[(slice(None), i), j].iloc[-1]
        total_portfolio_variance += stock_i * stock_j * cov_ij

# Calculate total VaR
total_VaR = z_score * (total_portfolio_variance ** 0.5)

# Print VaR result using an exponentially weighted covariance
print("\nPortfolio Value at Risk (VaR):")
for portfolio, VaR in portfolio_VaRs.items():
    print("{:}: ${:.4f}".format(portfolio, VaR))

print("\nTotal VaR of the Portfolio:")
print('${:.4f}'.format(total_VaR))


# Calculate historical VaR
def calculate_historical_var(returns, confidence_level=0.95):
    sorted_returns = returns.sort_values()
    index = int((1-confidence_level) * len(sorted_returns))
    VaR = sorted_returns.iloc[index]
    return VaR

portfolio_returns = {}

for portfolio_name, portfolio_data in portfolio_df.groupby('Portfolio'):
    portfolio_daily_returns = pd.Series(0, index=daily_returns_df.index)
    for _, row in portfolio_data.iterrows():
        stock_returns = daily_returns_df[row['Stock']]
        weighted_returns = stock_returns * row['Holding']
        portfolio_daily_returns += weighted_returns
    portfolio_returns[portfolio_name] = portfolio_daily_returns

# Step 4: Calculate Historical VaR for Each Portfolio
portfolio_VaRs = {}

for portfolio_name, returns in portfolio_returns.items():
    VaR = calculate_historical_var(returns, confidence_level=0.95)
    portfolio_VaRs[portfolio_name] = VaR

# Step 5: Calculate Historical VaR for Total Holdings
total_holdings_returns = pd.Series(0, index=daily_returns_df.index)
for _, row in portfolio_df.iterrows():
    stock_returns = daily_returns_df[row['Stock']]
    weighted_returns = stock_returns * row['Holding']
    total_holdings_returns += weighted_returns

total_VaR = calculate_historical_var(total_holdings_returns, confidence_level=0.95)

# Display the results
print("\nPortfolio Value at Risk (VaR):")
for portfolio, VaR in portfolio_VaRs.items():
    print(f"{portfolio}: ${-VaR:.4f}")

print("\nTotal VaR of the Portfolio:")
print(f"${-total_VaR:.4f}")
