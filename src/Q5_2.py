import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.special import comb
from scipy.stats import t as student_t

def calculate_p_value(t_statistic, df):
    return 2 * (1 - student_t.cdf(abs(t_statistic), df))

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
    if distribution == 'weibull':
        random_vars = np.random.weibull(shape, n) * scale
    elif distribution == 'pareto':
        random_vars = np.random.pareto(shape, n) * scale
    else:
        raise ValueError("Distribution must be 'weibull' or 'pareto'")

    pairwise_mins = np.minimum.outer(random_vars, random_vars)
    min_sum = np.sum(pairwise_mins[np.triu_indices(n, k=1)])
    combinations = comb(n, 2, exact=True)
    result = min_sum / combinations
    return result

def calculate_s2_function(n, distribution, shape, scale):
    if distribution == 'weibull':
        random_vars = np.random.weibull(shape, n) * scale
    elif distribution == 'pareto':
        random_vars = np.random.pareto(shape, n) * scale
    else:
        raise ValueError("Distribution must be 'weibull' or 'pareto'")

    pairwise_mins = np.minimum.outer(random_vars, random_vars)
    min_sum = np.sum(pairwise_mins[np.triu_indices(n, k=1)])
    result = min_sum / (n-1)
    final_var = ((result - calculate_min_sum_same_distribution(n, distribution, shape, scale))**2) / (n-1)
    return final_var

# Define parameters
n = 30
m = 32
distribution = 'pareto'
distribution2 = 'pareto'
shape = 10
shape2 = 10
scale = 2

# Arrays to store scale2 and p-value of t_xy values
scale2_values = np.arange(1, 1001)
p_values = []

# Loop over scale2 values
for scale2 in scale2_values:
    scale2 = float(scale2)
    delta_xy = (calculate_min_sum_same_distribution(n, distribution, shape, scale) / calculate_average_random_distribution(n, distribution, shape, scale)) - (calculate_min_sum_same_distribution(m, distribution2, shape2, scale2) / calculate_average_random_distribution(m, distribution2, shape2, scale2))
    s2_xy = ((m / (n + m)) * (calculate_s2_function(n, distribution, shape, scale) / (calculate_average_random_distribution(n, distribution, shape, scale))**2)) + ((n / (m + n)) * (calculate_s2_function(m, distribution2, shape2, scale2) / (calculate_average_random_distribution(m, distribution2, shape2, scale2))**2))
    t = sqrt(n * m / (n + m)) * delta_xy / sqrt(s2_xy)
    df = n + m - 2
    p_value = calculate_p_value(t, df)
    p_values.append(p_value)

# Plotting as a heatmap
plt.figure(figsize=(10, 6))
plt.hist2d(scale2_values, p_values, bins=100, cmap='viridis')
plt.colorbar(label='Frequency')
plt.xlabel('Scale2', fontsize=14)
plt.ylabel('p-value', fontsize=14)
plt.title('2D Histogram: Scale2 vs p-value of |t_xy|', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
