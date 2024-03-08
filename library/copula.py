import pandas as pd
import numpy as np
from scipy.stats import norm, t
from simulate import simulate_pca

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