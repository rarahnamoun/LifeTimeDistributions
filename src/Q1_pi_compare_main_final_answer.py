import math
import random
import matplotlib.pyplot as plt

from scipy.stats import expon, pareto
from scipy.integrate import quad

def pareto_reliability(k, r, alpha):
    t = alpha / (alpha - 1)
    numerator = math.factorial(k) / math.factorial(k - r)
    denominator = math.gamma(k - r + 1 + t) / math.gamma(k + 1 + t)
    return numerator * denominator

def exponential_reliability(k, r):
    return (k - r + 1) / (k + 1)

def inverse_expectation(distribution_function, t, max_limit=1000):
    def integrand(x):
        return 1 - distribution_function(x)

    integral_value, _ = quad(integrand, t, max_limit)
    return integral_value / (1 - distribution_function(t))

def inverse_pareto_distribution(p, b, scale):
    return pareto.cdf(1/p, b, scale=scale)

def inverse_exponential_distribution(p, scale):
    return expon.cdf(1/p, scale=scale)

def pareto_distribution(p, b, scale):
    return pareto.cdf(p, b, scale=scale)

def exponential_distribution(p, scale):
    return expon.cdf(p, scale=scale)

def compare_expectations(first_inv_exp1, second_inv_exp1, first_inv_exp2, second_inv_exp2, t, max_limit=1000):
    ratio1 = first_inv_exp1 / second_inv_exp1
    ratio2 = first_inv_exp2 / second_inv_exp2

    if ratio1 >= ratio2:
        return "NWUE"
    else:
        return "NBUE"

# Example usage with a Gaussian (Normal) Distribution
t = 50  # Choose any value of t

# Number of iterations
num_iterations = 1000

# Lists to store the results for plotting
categories = ["NBUE", "NWUE"]
counts = [0, 0]

# Lists to store the Pareto and Exponential reliabilities
pareto_results = []
exponential_results = []

# Perform comparison in a while loop with random parameters
for _ in range(num_iterations):
    # Generate random parameters
    b_pareto = random.uniform(1, 10)  # Random shape parameter for Pareto distribution
    scale_pareto = random.uniform(50, 500)  # Random scale parameter for Pareto distribution
    scale_exponential = random.uniform(500, 800)  # Random scale parameter for Exponential distribution

    result = compare_expectations(
        inverse_expectation(lambda x: inverse_pareto_distribution(x, b_pareto, scale_pareto), t),
        inverse_expectation(lambda x: inverse_exponential_distribution(x, scale_exponential), t),
        inverse_expectation(lambda x: pareto_distribution(x, b_pareto, scale_pareto), t),
        inverse_expectation(lambda x: exponential_distribution(x, scale_exponential), t),
        t=t,
        max_limit=1000
    )

    if result == "NBUE":
        counts[0] += 1
    elif result == "NWUE":
        counts[1] += 1

    # Calculate reliabilities for Pareto and Exponential distributions
    pareto_result = pareto_reliability(100, 10, scale_pareto)
    exponential_result = exponential_reliability(100, 10)

    pareto_results.append(pareto_result)
    exponential_results.append(exponential_result)

# Plot Pareto and Exponential reliabilities
plt.plot(range(1, num_iterations + 1), pareto_results, label='Pareto Reliability')
plt.plot(range(1, num_iterations + 1), [exponential_result]*num_iterations, label='Exponential Reliability')

# Add text annotations for counts of NBUE and NWUE
plt.text(num_iterations, max(max(pareto_results), exponential_result), f"NWUE: {counts[1]}/{num_iterations}", ha='right', va='bottom', fontsize=10, color='red')


plt.xlabel('Iteration')
plt.ylabel('Reliability')
plt.title('Pareto vs. Exponential Reliability')
plt.legend()
plt.show()
