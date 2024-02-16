import random
import matplotlib.pyplot as plt

from scipy.stats import expon, pareto
from scipy.integrate import quad

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
num_iterations = 100

# Lists to store the results for plotting
categories = ["NBUE", "NWUE"]
counts = [0, 0]

# Perform comparison in a while loop with random parameters
for _ in range(num_iterations):
    # Generate random parameters
    b_pareto = random.uniform(1, 10)  # Random shape parameter for Pareto distribution
    scale_pareto = random.uniform(1, 20)  # Random scale parameter for Pareto distribution
    scale_exponential = random.uniform(200, 500)  # Random scale parameter for Exponential distribution

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

# Plot the results with customized styling
plt.figure(figsize=(8, 6))
bars = plt.bar(categories, counts, color=['skyblue', 'lightcoral'])

# Add data labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Add title and labels
plt.title("Comparison of NBUE and NWUE ", fontsize=16)

plt.ylabel("Count", fontsize=14)

# Add grid lines
plt.grid(True, axis='y', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()
