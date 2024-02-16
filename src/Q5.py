import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import comb

def calculate_average_random_distribution(n, distribution, shape, scale):
    if distribution == 'weibull':
        random_vars = np.random.weibull(shape, n) * scale
    elif distribution == 'pareto':
        random_vars = np.random.pareto(shape, n) * scale
    sum_results = sum(random_vars)

    # Calculate the average
    average_result = sum_results / n
    return average_result

def calculate_min_sum_same_distribution(n, distribution, shape, scale):
    # Generate n random variables from the given distribution
    if distribution == 'weibull':
        random_vars = np.random.weibull(shape, n) * scale
    elif distribution == 'pareto':
        random_vars = np.random.pareto(shape, n) * scale
    else:
        raise ValueError("Distribution must be 'weibull' or 'pareto'")

    # Calculate the pairwise minimums
    pairwise_mins = np.minimum.outer(random_vars, random_vars)

    # Calculate the sum of the pairwise minimums excluding diagonal elements
    min_sum = np.sum(pairwise_mins[np.triu_indices(n, k=1)])

    # Calculate C(n, 2)
    combinations = comb(n, 2, exact=True)

    # Calculate the final result
    result = min_sum / combinations
    return result

def calculate_s2_function(n, distribution, shape, scale):
    # Generate n random variables from the given distribution
    if distribution == 'weibull':
        random_vars = np.random.weibull(shape, n) * scale
    elif distribution == 'pareto':
        random_vars = np.random.pareto(shape, n) * scale
    else:
        raise ValueError("Distribution must be 'weibull' or 'pareto'")

    # Calculate the pairwise minimums
    pairwise_mins = np.minimum.outer(random_vars, random_vars)

    # Calculate the sum of the pairwise minimums excluding diagonal elements
    min_sum = np.sum(pairwise_mins[np.triu_indices(n, k=1)])

    # Calculate the final result
    result = min_sum / (n-1)
    final_var = ((result - calculate_min_sum_same_distribution(n, distribution, shape, scale))**2) / (n-1)

    return final_var

def z_test(t_input_var, alpha):
    # Calculate the critical value (Z(1-alpha))
    z_critical = stats.norm.ppf(1 - alpha)
    p_value = (1 - stats.norm.cdf(t_input_var))
    return p_value

# Parameters
n = 500
m = 520
distribution = 'pareto'
distribution2 = 'pareto'
shape = 10
shape2 = 10
alpha = 0.05

# Arrays to store averages
average_values = []

# Scale range
scales = range(1, 48)

# Calculate average for each scale
for scale in scales:
    # Arrays to store results for this scale
    results = []

    # Run the calculations 100 times
    for _ in range(50):
        delta_xy = (calculate_min_sum_same_distribution(n, distribution, shape, scale) / calculate_average_random_distribution(n, distribution, shape, scale)) - (calculate_min_sum_same_distribution(m, distribution2, shape2, scale) / calculate_average_random_distribution(m, distribution2, shape2, scale))
        s2_xy = ((m / (n + m)) * ((calculate_s2_function(n, distribution, shape, scale) / scale) / (calculate_average_random_distribution(n, distribution, shape, scale))**2)) + (((n / (m + n)) * (calculate_s2_function(m, distribution2, shape2, scale) / scale) / (calculate_average_random_distribution(m, distribution2, shape2, scale))**2))
        t_xy = (np.sqrt(n * m / (n + m)) * delta_xy) / (np.sqrt(s2_xy))
        p_value = z_test(abs(t_xy), alpha)
        results.append(p_value)

    # Calculate the average of the 100 results
    average_value = np.mean(results)
    average_values.append(average_value)

# Plot
plt.plot(scales, average_values)
plt.title('Average P-value vs Scale')
plt.xlabel('Scale')
plt.ylabel('Average P-value')
plt.grid(True)
plt.show()
