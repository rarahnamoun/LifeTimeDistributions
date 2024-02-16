import math
import numpy as np
import matplotlib.pyplot as plt

def pareto_reliability(k, r, alpha):
    t = alpha / (alpha - 1)
    numerator = math.factorial(k) / math.factorial(k - r)
    denominator = math.gamma(k - r + 1 + t) / math.gamma(k + 1 + t)
    return numerator * denominator

def uniform_reliability(k, r):
    numerator = math.factorial(k) / math.factorial(k - r)
    denominator = math.gamma(k - r + 1.5) / math.gamma(k + 1.5)
    return numerator * denominator

def exponential_reliability(k, r):
    return (k - r + 1) / (k + 1)

# Generating k and r values such that r is always smaller than k
k_values = np.arange(10, 101, 5)
r_values = np.arange(5, 96, 5)  # r must always be smaller than k
k_mesh, r_mesh = np.meshgrid(k_values, r_values)
pareto_results = np.zeros_like(k_mesh, dtype=float)
uniform_results = np.zeros_like(k_mesh, dtype=float)
exponential_results = np.zeros_like(k_mesh, dtype=float)

for i in range(len(k_values)):
    for j in range(len(r_values)):
        if r_values[j] < k_values[i]:
            pareto_results[j, i] = pareto_reliability(k_values[i], r_values[j], alpha=5)
            uniform_results[j, i] = uniform_reliability(k_values[i], r_values[j])
            exponential_results[j, i] = exponential_reliability(k_values[i], r_values[j])

fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(k_mesh, r_mesh, pareto_results, cmap='viridis')
ax1.set_title('Pareto Isolation')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(k_mesh, r_mesh, uniform_results, cmap='viridis')
ax2.set_title('Uniform Isolation')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(k_mesh, r_mesh, exponential_results, cmap='viridis')
ax3.set_title('Exponential Isolation')

plt.show()
