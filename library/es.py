import pandas as pd
import numpy as np
from scipy.stats import t, norm
from fitted_model import fit_normal, fit_general_t
from var import var_normal, var_t
from scipy.integrate import quad
    
#ES for normal distribution
def es_normal(data, alpha=0.05):
    #Fit the data with normal distribution.
    mu, std = fit_normal(data)

    res = var_normal(data, alpha)
    VaR = res.iloc[0, 0]
    #Define the integrand function: x times the PDF of the distribution
    def integrand(x, mu, std):
        return x * norm.pdf(x, loc=mu, scale=std)
    
    ES, _ = quad(lambda x: integrand(x, mu, std), -np.inf, -VaR)
    ES /= -alpha
    #Calculate the relative difference from the mean expected.
    ES_diff = ES + mu
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [ES_diff]})

#ES for t Distribution
def es_t(data, alpha=0.05):
    #Fit the data with normal distribution.
    mu, sigma, nu = fit_general_t(data)
    
    res = var_t(data, alpha)
    VaR = res.iloc[0, 0]
    #Define the integrand function: x times the PDF of the distribution
    def integrand(x, mu, sigma, nu):
        return x * t.pdf(x, df=nu, loc=mu, scale=sigma)

    ES, _ = quad(lambda x: integrand(x, mu, sigma, nu), -np.inf, -VaR)
    ES /= -alpha
    ES_diff = ES + mu
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [ES_diff]})

#ES for simulation
def es_simulation(data, alpha=0.05, size=10000):
    #Fit the data with t distribution.
    mu, sigma, nu = fit_general_t(data)
    random_numbers = t.rvs(df=nu, loc=mu, scale=sigma, size=size)
    return es_t(random_numbers, alpha)