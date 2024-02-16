import scipy.stats as stats
import scipy.special as sp
from scipy.integrate import quad

# Define the complementary cumulative distribution functions for different distributions
def complementary_cdf(x, distribution, *args, **kwargs):
    if distribution == 'weibull':
        return  1-stats.weibull_min.cdf(x, *args, **kwargs)
    elif distribution == 'levy':
        return  1-stats.levy.cdf(x, *args, **kwargs)
    elif distribution == 'lognorm':
        return  1-stats.lognorm.cdf(x, *args, **kwargs)
    elif distribution == 'gamma':
        return  1-stats.gamma.cdf(x, *args, **kwargs)
    elif distribution == 'invgauss':
        return  1-stats.invgauss.cdf(x, *args, **kwargs)
    else:
        raise ValueError("Invalid distribution specified.")

# Define the equation function for different distributions
def calculate_equation(r, k, distribution, *args, **kwargs):
    # Calculate the expected value of the distribution function raised to the power of k
    expected_value_fk = quad(lambda x: complementary_cdf(x, distribution, *args, **kwargs) ** k, 0, float('inf'))[0]

    # Calculate the integral with nested integrals
    def integrand_t(t):
        inner_integral1 = quad(lambda u: 1 - complementary_cdf(u, distribution, *args, **kwargs), 0, t)[0]
        if inner_integral1 != 0:
            inner_integral2 = quad(lambda u: 1 - complementary_cdf(u, distribution, *args, **kwargs), t, t + 100)[0]
            return (1 - complementary_cdf(t, distribution, *args, **kwargs)) ** 2 * inner_integral1 ** (r - 1) * inner_integral2 ** (k - r)

    integral_value, _ = quad(integrand_t, 5, float('inf'))

    # Calculate the equation
    equation_value = (r / expected_value_fk) * sp.comb(k, r) * integral_value
    return equation_value

# Parameters
r = 20
k = 50
distribution = 'levy'  # Specify the distribution ('weibull', 'levy', 'lognorm', 'gamma', 'invgauss')
distribution_params = {  # Distribution parameters
    'weibull': (1.5, 0, 1),     # Shape, loc, scale for Weibull distribution
    'levy': (0, 5),             # loc, scale for Levy distribution
    'lognorm': (0.5, 0, 1),     # s (shape), loc, scale for Log-normal distribution
    'gamma': (2, 0, 1),         # a (shape), loc, scale for Gamma distribution
    'invgauss': (1, 0, 1)       # mu (mean), loc, scale for Inverse Gaussian distribution
}

# Calculate the equation
result = calculate_equation(r, k, distribution, *distribution_params[distribution])
print("Result:", result)