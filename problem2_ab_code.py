import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t
from scipy import stats

# Load the data from the CSV file
file_path = '/Users/heyahe/FINTECH-545/Week02/problem2.csv'
data = pd.read_csv(file_path)

X_ols = sm.add_constant(data['x'])
y_ols = data['y']

# Create the OLS Model
ols_model = sm.OLS(y_ols, X_ols)
ols_results = ols_model.fit()
ols_coefficients = ols_results.params[1:].values[0]
ols_intercept = ols_results.params[0]
ols_std_dev = ols_results.mse_resid ** 0.5

# Display the summary of the OLS model
ols_model_summary = ols_results.summary()
print(ols_model_summary)
print("\nThe slope coefficient from OLS model is:", ols_coefficients)
print("The intercept from OLS model is:", ols_intercept)
print("The standard deviation of the OLS error is:", ols_std_dev)

X_mle = data['x']
Y_mle = data['y']

# Define the likelihood function
def normal_log_likelihood(beta, x, y):
    beta0, beta1, sigma = beta
    y_pred = beta0 + beta1 * x
    ll = -np.sum(norm.logpdf(y, loc=y_pred, scale=sigma))
    return ll

# Optimize the likelihood function
mlen_result = minimize(normal_log_likelihood, [0, 0, 1], args=(X_mle, Y_mle)).x
mlen_coefficients = mlen_result[1]
mlen_intercept = mlen_result[0]
mlen_std_dev = mlen_result[2]

# Print the results
print("\nHere is the results using MLE given the assumption of normality")
print(f"Estimated Parameters: {mlen_result}")
print("The slope coefficient from MLE model is:", mlen_coefficients)
print("The intercept from MLE model is:", mlen_intercept)
print("The standard deviation of the MLE error is:", mlen_std_dev)

# Define the log-likelihood function for the T distribution
def t_log_likelihood(params, x, y):
    beta1, beta0, sigma, nu = params
    y_pred = beta0 + beta1 * x
    ll = np.log(stats.t.pdf(y, df=nu, loc=y_pred, scale=sigma))
    return -np.sum(ll)

# Optimize the likelihood function
mlet_result = minimize(t_log_likelihood, [0, 0, 1, 2], args=(X_mle, Y_mle), method='L-BFGS-B').x
mlet_coefficients = mlet_result[1]
mlet_intercept = mlet_result[0]
mlet_std_dev = mlet_result[2]
mlet_df = mlet_result[3]

# Print the results
print("\nHere is the results using MLE given the assumption of a T distribution")
print(f"Estimated Parameters: {mlet_result}")
print("The slope coefficient from MLE model is:", mlet_coefficients)
print("The intercept from MLE model is:", mlet_intercept)
print("The standard deviation of the MLE error is:", mlet_std_dev)
print("The degrees of freedom of the T-distribution is:", mlet_df)

# Calculate AIC for normal distribution
k_normal = len(mlen_result)
max_log_likelihood_normal = -normal_log_likelihood(mlen_result, X_mle, Y_mle)
aic_normal = 2 * k_normal - 2 * max_log_likelihood_normal

# Calculate AIC for T distribution
k_t = len(mlet_result)
max_log_likelihood_t = -t_log_likelihood(mlet_result, X_mle, Y_mle)
aic_t = 2 * k_t - 2 * max_log_likelihood_t

# Print the AIC values
print("\nAIC for normal distribution model:", aic_normal)
print("AIC for T distribution model:", aic_t)
