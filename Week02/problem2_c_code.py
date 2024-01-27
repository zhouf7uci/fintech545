import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt

# Load the data from the CSV files
file_x = '/Users/heyahe/FINTECH-545/Week02/problem2_x.csv'
file_x1 = '/Users/heyahe/FINTECH-545/Week02/problem2_x1.csv'

data_x = pd.read_csv(file_x)
data_x1 = pd.read_csv(file_x1)

# Calculate the mean and covariance matrix using Maximum Likelihood Estimation and print
mean = np.mean(data_x, axis=0)
covariance = np.cov(data_x, rowvar=False)
print(mean)
print(covariance)

# Create the multivariate normal distribution
mvn_distribution = multivariate_normal(mean=mean, cov=covariance)
mu_x1, mu_x2 = mean
sigma_x1x1, sigma_x2x1 = covariance[0, 0], covariance[1, 0]
sigma_x1x2, sigma_x2x2 = covariance[0, 1], covariance[1, 1]

# Calculate conditional means and variance
conditional_means = mu_x2 + (sigma_x2x1 / sigma_x1x1) * (data_x1['x1'] - mu_x1)
conditional_variance = sigma_x2x2 - (sigma_x2x1**2 / sigma_x1x1)
conditional_std = np.sqrt(conditional_variance)

# Calculate the 95% confidence interval
z = norm.ppf(0.975)
lower_bound = conditional_means - z * conditional_std
upper_bound = conditional_means + z * conditional_std

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data_x1['x1'], conditional_means, label='Expected Value of X2')
plt.fill_between(data_x1['x1'], lower_bound, upper_bound, color='gray', alpha=0.5, label='95% Confidence Interval')
plt.xlabel('Observed X1')
plt.ylabel('Conditional Distribution of X2')
plt.title('Expected Value and 95% Confidence Interval of X2 Given X1')
plt.legend()
plt.show()
