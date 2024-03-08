import pandas as pd
from scipy.stats import t, norm
from fitted_model import fit_normal, fit_general_t

def var_normal(data, alpha=0.05):
    #Fit the data with normal distribution.
    mu, std = fit_normal(data)
    VaR = -norm.ppf(alpha, mu, std)
    #Calculate the relative difference from the mean expected.
    VaR_diff = VaR + mu
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [VaR_diff]})

def var_t(data, alpha=0.05):
    #Fit the data with t distribution.
    mu, sigma, nu = fit_general_t(data)
    VaR = -t.ppf(alpha, nu, mu, sigma)
    #From the mean expected.
    VaR_diff = VaR + mu
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [VaR_diff]})

#VaR for t Distribution simulation 
def var_simulation(data, alpha=0.05, size=10000):
    #Fit the data with t distribution.
    mu, sigma, nu = fit_general_t(data)
    #Generate given size random numbers from a t-distribution
    random_numbers = t.rvs(df=nu, loc=mu, scale=sigma, size=size)
    return var_t(random_numbers, alpha)
