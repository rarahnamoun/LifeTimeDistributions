import matplotlib.pyplot as plt
import numpy as np
import math

def pareto_reliability(k, r, alpha):
    t = alpha / (alpha - 1)
    numerator = math.factorial(k) / math.factorial(k - r)
    denominator = math.gamma(k - r + 1 + t) / math.gamma(k + 1 + t)
    return numerator * denominator

# Define parameters
k = 10
r = 5
alphas = np.arange(1, 101)

# Calculate reliability values
reliability_values = [pareto_reliability(k, r, alpha) for alpha in alphas]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(alphas, reliability_values, color='blue', linewidth=2, label='Pareto Reliability')
plt.xlabel('Alpha', fontsize=14)
plt.ylabel('Isolation', fontsize=14)
plt.title('Pareto Isolation vs Alpha', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()
