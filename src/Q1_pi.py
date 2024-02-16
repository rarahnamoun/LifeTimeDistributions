import scipy.stats as stats
import scipy.special as sp
from scipy.integrate import quad

# Define the complementary cumulative distribution functions for Pareto and Exponential distributions
def pareto_complementary_cdf(x, b, scale):
    return 1- stats.pareto.cdf(x, b, scale=scale)



# Define the equation function
def calculate_equation(r, k, b_pareto, scale_pareto):
    # Calculate the expected value of the distribution function raised to the power of k
    expected_value_fk = quad(lambda x: (pareto_complementary_cdf(x, b_pareto, scale_pareto)) ** k, 0, float('inf'))[0]
    # Calculate the integral with nested integrals
    def integrand_t(t):
        inner_integral1 = quad(lambda u: 1 - pareto_complementary_cdf(u, b_pareto, scale_pareto), 0, t)[0]
        if (inner_integral1 != 0):
            inner_integral2 = quad(lambda u: 1 - pareto_complementary_cdf(u, b_pareto, scale_pareto), t, t + 100)[0]
            return (1 - pareto_complementary_cdf(t, b_pareto, scale_pareto)) ** 2 * inner_integral1 ** (r - 1) * inner_integral2 ** (k - r)

    integral_value, _ = quad(integrand_t, 5, float('inf'))

    # Calculate the equation
    equation_value = (r / expected_value_fk) * sp.comb(k, r) * integral_value
    return equation_value

# Parameters
r = 20
k = 50
b_pareto = 3  # Shape parameter for Pareto distribution
scale_pareto = 4  # Scale parameter for Pareto distribution


# Calculate the equation
result = calculate_equation(r, k, b_pareto, scale_pareto)
print("Result:", result)
