import pandas as pd
import numpy as np
from scipy.stats import t, norm
import statsmodels.api as sm
from scipy.optimize import minimize

def fit_normal(data):
    # Fit the normal distribution to the data
    mu, std = norm.fit(data)
    return mu, std

def fit_general_t(data):
    # Fit the t distribution to the data
    nu, mu, sigma = t.fit(data)
    return mu, sigma, nu

#Fit t distribution
def fit_regression_t(df):
    Y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    betas = MLE_t(X, Y)
    X = sm.add_constant(X)
    
    # Get the residuals.
    e = Y - np.dot(X, betas)

    params = t.fit(e)
    out = {"mu": [params[1]], 
           "sigma": [params[2]], 
           "nu": [params[0]]}
    for i in range(len(betas)):
        out["B" + str(i)] = betas[i]
    out = pd.DataFrame(out)
    out.rename(columns={'B0': 'Alpha'}, inplace=True)
    return out

#The objective negative log-likelihood function (need to be minimized).
def MLE_t(X, Y):
    X = sm.add_constant(X)
    def ll_t(params):
        nu, sigma = params[:2]
        beta_MLE_t = params[2:]
        epsilon = Y - np.dot(X, beta_MLE_t)
        # Calculate the log-likelihood
        log_likelihood = np.sum(t.logpdf(epsilon, df=nu, loc=mu, scale=sigma))
        return -log_likelihood
    
    beta = np.zeros(X.shape[1])
    nu, mu, sigma = 1, 0, np.std(Y - np.dot(X, beta))
    params = np.append([nu, sigma], beta)
    bnds = ((0, None), (0, None), (None, None), (None, None), (None, None), (None, None))
    
    # Minimize the log-likelihood to get the beta
    res = minimize(ll_t, params, bounds=bnds, options={'disp': True})
    beta_MLE = res.x[2:]
    return beta_MLE
