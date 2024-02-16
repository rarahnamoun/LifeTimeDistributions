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
    result = (k - r + 1) / (k + 1)
    return result

# Example usage:
k = 100
alpha = 5
r_values = range(5, 50)  # r varies from 5 to 500

pareto_results = []
uniform_results = []
exponential_results = []

for r in r_values:
    pareto_result = pareto_reliability(k, r, alpha)
    uniform_result = uniform_reliability(k, r)
    exponential_result = exponential_reliability(k, r)

    pareto_results.append(pareto_result)
    uniform_results.append(uniform_result)
    exponential_results.append(exponential_result)

plt.plot(r_values, pareto_results, label='Pareto Reliability')
plt.plot(r_values, uniform_results, label='Uniform Reliability')
plt.plot(r_values, exponential_results, label='Exponential Reliability')

plt.xlabel('r')
plt.ylabel('Reliability')
plt.title('Reliability vs. r for k=100')
plt.legend()
plt.show()
