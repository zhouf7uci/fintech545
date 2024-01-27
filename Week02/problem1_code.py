import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, ttest_1samp

# Load the data from the CSV file
file_path = '/Users/heyahe/FINTECH-545/Week02/problem1.csv'
data = pd.read_csv(file_path)
sim = data['x'].values.reshape(-1, 1)

# Calculate the first four moments values by using normalized formula
def first4Moments(sample):
    n = sample.shape[0]
    mu_hat = np.mean(sample)
    sim_corrected = sample - mu_hat
    cm2 = np.dot(sim_corrected.T, sim_corrected) / n
    sigma2_hat = np.dot(sim_corrected.T, sim_corrected) / (n - 1)
    skew_hat = np.sum(sim_corrected ** 3) / n / (cm2 ** 1.5)
    kurt_hat = np.sum(sim_corrected ** 4) / n / (cm2 ** 2)
    excessKurt_hat = kurt_hat - 3
    return mu_hat, sigma2_hat[0,0], skew_hat[0,0], excessKurt_hat[0,0]

m, s2, sk, k = first4Moments(sim)

# Print the results
print("First four moments values by using normalized formula")
print("Mean:", m)
print("Variance:", s2)
print("Skewness:", sk)
print("Kurtosis:", k)

m_sp = np.mean(sim)
s2_sp = np.var(sim, ddof=1)
sk_sp = skew(sim)
k_sp = kurtosis(sim, fisher=True)
print("\nFirst four moments values by using statistical package")
print("Mean:", m_sp)
print("Variance:", s2_sp)
print("Skewness:", sk_sp[0])
print("Kurtosis:", k_sp[0])

# Function to create synthetic data with known properties
def create_synthetic_data(n, mean, variance, skewness, kurtosis):
    data = np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
    return data

# Number of datasets and samples per dataset
num_datasets = 100
n_samples = 1000

# Known properties
known_mean = 0
known_variance = 1
known_skewness = 0
known_kurtosis = 3

# Store the differences between calculated and true values
differences = {
    'mean': [],
    'variance': [],
    'skewness': [],
    'kurtosis': []
}

# Generate synthetic datasets and calculate moments
for i in range(num_datasets):
    data = create_synthetic_data(n_samples, known_mean, known_variance, known_skewness, known_kurtosis).reshape(-1, 1)
    m, s2, sk, k = first4Moments(data)
    m_sp = np.mean(data)
    s2_sp = np.var(data, ddof=1)
    sk_sp = skew(data)[0]
    k_sp = kurtosis(data, fisher=True)[0]

    # Store the differences
    differences['mean'].append(m - m_sp)
    differences['variance'].append(s2 - s2_sp)
    differences['skewness'].append(sk - sk_sp)
    differences['kurtosis'].append(k - k_sp)

# Perform t-tests to check if differences are statistically significant
t_test_results = {}
for moment in differences:
    t_test_results[moment] = ttest_1samp(differences[moment], 0)
print()
print(t_test_results)
