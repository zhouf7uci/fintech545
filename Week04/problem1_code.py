import numpy as np
import math

# Define the parameters P(t-1)=100, sigma=0.02 for the simulation
sigma = 0.02
P_0 = 100

# Simulate prices with 10000 times
sim_returns = np.random.normal(0,sigma,10000)

# Classic Brownian Motion: P_t = P_(t-1) + r_t
price_brownian = P_0 + sim_returns

# Calculate mean and standard deviation of sim_prices
sim_mean_brownian = price_brownian.mean()
sim_std_brownian = price_brownian.std()

# Theoretical mean and standard deviation of P(t) in Classic Brownian Motion
theory_mean_brownian = P_0
theory_std_brownian = sigma

# Arithmetic Return System: P_t = P_(t-1) * (1 + r_t)
price_arithmetic = P_0 * (1 + sim_returns)

# Calculate mean and standard deviation of sim_prices
sim_mean_arithmetic = price_arithmetic.mean()
sim_std_arithmetic = price_arithmetic.std()

# Theoretical mean and standard deviation of P(t) in Arithmetic Return System
theory_mean_arithmetic = P_0
theory_std_arithmetic = P_0 * sigma

# Log Return or Geometric Brownian Motion: P_t = P_(t-1) * e^(r_t)
price_log = P_0 * np.exp(sim_returns)

# Calculate mean and standard deviation of sim_prices
sim_mean_log = price_log.mean()
sim_std_log = price_log.std()

# Theoretical mean and standard deviation of P(t) in Log Return
theory_mean_log = P_0 * math.exp(0.5 * pow(sigma,2))
theory_std_log = P_0 * math.sqrt(math.exp(pow(sigma,2)) - 1)

# Print the expected value for each model
print(f"Expected value for Classical Brownian Motion: {theory_mean_brownian:.2f}")
print(f"Expected value for Arithmetic Return System: {theory_mean_arithmetic:.2f}")
print(f"Expected value for Log Return or Geometric Brownian Motion: {theory_mean_log:.2f}\n")

# Print the expected standard deviation for each model
print(f"Expected standard deviation for Classical Brownian Motion: {theory_std_brownian:.2f}")
print(f"Expected standard deviation for Arithmetic Return System: {theory_std_arithmetic:.2f}")
print(f"Expected standard deviation for Log Return or Geometric Brownian Motion: {theory_std_log:.2f}\n")

# Print the simulated prices for each model
print(f"Simulated price using Classical Brownian Motion: {sim_mean_brownian:.2f}")
print(f"Simulated price using Arithmetic Return System: {sim_mean_arithmetic:.2f}")
print(f"Simulated price using Log Return or Geometric Brownian Motion: {sim_mean_log:.2f}\n")

# Print the simulated standard deviation for each model
print(f"Simulated standard deviation using Classical Brownian Motion: {sim_std_brownian:.2f}")
print(f"Simulated standard deviation using Arithmetic Return System: {sim_std_arithmetic:.2f}")
print(f"Simulated standard deviation using Log Return or Geometric Brownian Motion: {sim_std_log:.2f}\n")
