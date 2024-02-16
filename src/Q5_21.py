

from cmath import sqrt
import scipy.stats as stats
import numpy as np
from scipy.special import comb
def calculate_average_random_distribution( n, distribution, shape, scale):
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
    final_var=((result-calculate_min_sum_same_distribution(n, distribution, shape, scale))**2)/ (n-1)

    return final_var  # Add this line to return the calculated value


# Example usage:
n = 30
m=32
distribution = 'pareto'
distribution2 = 'pareto'
shape =10
scale = 2
shape2 = 10
scale2 = 1.25


delta_xy=(calculate_min_sum_same_distribution(n, distribution, shape, scale)/calculate_average_random_distribution( n, distribution, shape, scale))-(calculate_min_sum_same_distribution(m, distribution2, shape2, scale2)/calculate_average_random_distribution(m, distribution2, shape2, scale2))
s2_xy=((m/(n+m))*((calculate_s2_function(n, distribution, shape, scale)/scale)/(calculate_average_random_distribution( n, distribution, shape, scale))**2))+(((n/(m+n))*(calculate_s2_function(m, distribution2, shape2, scale2)/scale2)/(calculate_average_random_distribution( m, distribution2, shape2, scale2))**2))
t_xy=(sqrt(n*m/(n+m))*delta_xy)/(sqrt(s2_xy))




def z_test(t_input_var, alpha):
    # Calculate the critical value (Z(1-alpha))
    z_critical = stats.norm.ppf(1 - alpha)
    p_value = 1 - stats.norm.cdf(  t_input_var)
    print( p_value)

    # Perform the hypothesis test
    if p_value < z_critical:
        return True  # Reject the null hypothesis
    else:
        return False  # Fail to reject the null hypothesis


alpha = 0.05

reject_null = z_test(abs(t_xy), alpha)
if reject_null:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")