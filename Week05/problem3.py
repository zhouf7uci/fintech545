#Problem 3
import pandas as pd
import numpy as np
from scipy.stats import norm, t

# copy from library.return_calculate
def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    vars = [var for var in prices.columns if var != dateColumn]
    if not vars:
        raise ValueError(f"{dateColumn} not in DataFrame: {prices.columns}")

    p = np.matrix(prices[vars])
    n, m = p.shape
    p2 = np.empty((n-1, m))

    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]

    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in ('LOG','DISCRETE')")

    dates = prices[dateColumn][1:n].reset_index(drop=True)
    data = {vars[i]: p2[:, i].flatten() for i in range(m)}
    data[dateColumn] = dates

    out = pd.DataFrame(data)
    return out

# copy from library.simulate
def simulate_pca(a, nsim, pctExp=1, mean=None, seed_val=1234):
    m, n = a.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")
    out = np.zeros((nsim,n))

    # Set the mean
    if mean is None:
        mean = np.zeros(n)
    else:
        if len(mean) != n:
            raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)
    indices = np.argsort(vals)[::-1]
    vals = vals[indices]
    vecs = vecs[:, indices]

    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if pctExp < 1:
        nval = 0
        pct = 0.0
        for i in posv:
            pct += vals[i] / tv
            nval += 1
            if pct >= pctExp:
                break
        if nval < len(posv):
            posv = posv[:nval]

    vals = vals[posv]
    vecs = vecs[:, posv]
    B = vecs @ np.diag(np.sqrt(vals))

    np.random.seed(seed_val)
    rand_normals = np.random.normal(0.0, 1.0, size=(nsim, len(posv)))
    out = np.dot(rand_normals, B.T) + mean
    return out.T

# copy from library.copula
def simulate_copula(portfolio, returns):
    portfolio['CurrentValue'] = portfolio['Holding'] * portfolio['Starting Price']
    models = {}
    uniform = pd.DataFrame()
    standard_normal = pd.DataFrame()
    
    for stock in portfolio["Stock"]:
        if portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'Normal':
            models[stock] = norm.fit(returns[stock])
            mu, sigma = norm.fit(returns[stock])
            uniform[stock] = norm.cdf(returns[stock], loc=mu, scale=sigma)
            standard_normal[stock] = norm.ppf(uniform[stock])
        elif portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'T':
            models[stock] = t.fit(returns[stock])
            nu, mu, sigma = t.fit(returns[stock])
            uniform[stock] = t.cdf(returns[stock], df=nu, loc=mu, scale=sigma)
            standard_normal[stock] = norm.ppf(uniform[stock])
        
    spearman_corr_matrix = standard_normal.corr(method='spearman')
    
    simulate_time = 10000
    
    # Use the PCA to simulate the multivariate normal
    simulations = simulate_pca(spearman_corr_matrix, simulate_time)
    simulations = pd.DataFrame(simulations.T, columns=[stock for stock in portfolio["Stock"]])
    uni = norm.cdf(simulations)
    uni = pd.DataFrame(uni, columns=[stock for stock in portfolio["Stock"]])
    simulatedReturns = pd.DataFrame()
    for stock in portfolio["Stock"]:
        if portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'Normal':
            mu, sigma = models[stock]
            simulatedReturns[stock] = norm.ppf(uni[stock], loc=mu, scale=sigma)
        elif portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'T':
            nu, mu, sigma = models[stock]
            simulatedReturns[stock] = t.ppf(uni[stock], df=nu, loc=mu, scale=sigma)
    
    simulatedValue = pd.DataFrame()
    pnl = pd.DataFrame()

    for stock in portfolio["Stock"]:
        currentValue = portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0]
        simulatedValue[stock] = currentValue * (1 + simulatedReturns[stock])
        pnl[stock] = simulatedValue[stock] - currentValue
        
    risk = pd.DataFrame(columns = ["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])
    w = pd.DataFrame()

    for stock in pnl.columns:
        i = risk.shape[0]
        risk.loc[i, "Stock"] = stock
        risk.loc[i, "VaR95"] = -np.percentile(pnl[stock], 5)
        risk.loc[i, "VaR95_Pct"] = risk.loc[i, "VaR95"] / portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0]
        risk.loc[i, "ES95"] = -pnl[stock][pnl[stock] <= -risk.loc[i, "VaR95"]].mean()
        risk.loc[i, "ES95_Pct"] = risk.loc[i, "ES95"] / portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0]
        w.at['Weight', stock] = portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0] / portfolio['CurrentValue'].sum()
        
    # Calculate the total pnl
    pnl['Total'] = 0
    for stock in portfolio["Stock"]:
        pnl['Total'] += pnl[stock]
    
    i = risk.shape[0]
    risk.loc[i, "Stock"] = 'Total'
    risk.loc[i, "VaR95"] = -np.percentile(pnl['Total'], 5)
    risk.loc[i, "VaR95_Pct"] = risk.loc[i, "VaR95"] / portfolio['CurrentValue'].sum()
    risk.loc[i, "ES95"] = -pnl['Total'][pnl['Total'] <= -risk.loc[i, "VaR95"]].mean()
    risk.loc[i, "ES95_Pct"] = risk.loc[i, "ES95"] / portfolio['CurrentValue'].sum()

    return risk


prices = pd.read_csv('Week05/DailyPrices.csv')
returns = return_calculate(prices)
returns = returns - returns.mean()
portfolio = pd.read_csv('Week05/portfolio.csv')
portfolio.loc[portfolio['Portfolio'].isin(['A', 'B']), 'Distribution'] = 'T'
portfolio.loc[portfolio['Portfolio'] == 'C', 'Distribution'] = 'Normal'
for stock in portfolio["Stock"]:
    portfolio.loc[portfolio['Stock'] == stock, 'Starting Price'] = prices.iloc[-1][stock]

simulate_copula(portfolio, returns)

portfolio_a = portfolio.loc[portfolio["Portfolio"] == "A"].copy()
risk_a = simulate_copula(portfolio_a, returns)
print("The risk matrix for Portfolio A is:\n", risk_a)
print("\n")

portfolio_b = portfolio.loc[portfolio["Portfolio"] == "B"].copy()
risk_b = simulate_copula(portfolio_b, returns)
print("The risk matrix for Portfolio B is:\n", risk_b)
print("\n")

portfolio_c = portfolio.loc[portfolio["Portfolio"] == "C"].copy()
risk_c = simulate_copula(portfolio_c, returns)
print("The risk matrix for Portfolio C is:\n", risk_c)
print("\n")