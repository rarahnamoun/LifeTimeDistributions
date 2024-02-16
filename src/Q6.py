import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto, weibull_min, expon, uniform

# Define parameters
num_samples = 10000
shift_value = 1
k = 3  # Number of neighbors a node has
r_values = np.arange(1, k+1)  # Range of r values from 1 to k

# Generate samples for shifted Pareto distribution
pareto_shape = 2
pareto_samples = pareto.rvs(pareto_shape, size=num_samples) + shift_value

# Generate samples for shifted Weibull distribution
weibull_shape = 2
weibull_samples = weibull_min.rvs(weibull_shape,scale=1.68, size=num_samples) + shift_value

# Generate samples for shifted Exponential distribution
exponential_scale = 1
exponential_samples = expon.rvs(scale=exponential_scale, size=num_samples) + shift_value

# Generate samples for Uniform distribution
uniform_samples = uniform.rvs(loc=shift_value, scale=1, size=num_samples)

# Calculate the probability of node isolation for each distribution
pareto_isolation_prob = [(pareto_samples <= i).mean() for i in r_values]
weibull_isolation_prob = [(weibull_samples <= i).mean() for i in r_values]
exponential_isolation_prob = [(exponential_samples <= i).mean() for i in r_values]
uniform_isolation_prob = [(uniform_samples <= i).mean() for i in r_values]

# Plot the probability of node isolation for each distribution
plt.figure(figsize=(10, 6))
plt.plot(r_values, pareto_isolation_prob, label='Shifted Pareto')
plt.plot(r_values, weibull_isolation_prob, label='Shifted Weibull')
plt.plot(r_values, exponential_isolation_prob, label='Shifted Exponential')
plt.plot(r_values, uniform_isolation_prob, label='Uniform')
plt.xlabel('Number of Bad Neighbors (r)')
plt.ylabel('Probability of Node Isolation')
plt.title('Probability of Node Isolation for Different Lifetime Distributions')
plt.legend()
plt.grid(True)
plt.show()
