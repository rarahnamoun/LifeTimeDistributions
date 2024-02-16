import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, levy, lognorm, gamma, invgauss
from scipy.integrate import quad
import scipy.special as sp

def complementary_cdf(x, distribution, *args, **kwargs):
    if distribution == 'weibull':
        return  1 - weibull_min.cdf(x, *args, **kwargs)
    elif distribution == 'levy':
        return  1 - levy.cdf(x, *args, **kwargs)
    elif distribution == 'lognorm':
        return  1 - lognorm.cdf(x, *args, **kwargs)
    elif distribution == 'gamma':
        return  1 - gamma.cdf(x, *args, **kwargs)
    elif distribution == 'invgauss':
        return  1 - invgauss.cdf(x, *args, **kwargs)
    else:
        raise ValueError("Invalid distribution specified.")

def calculate_equation(r, k, distribution, *args, **kwargs):
    expected_value_fk = quad(lambda x: complementary_cdf(x, distribution, *args, **kwargs) ** k, 0, np.inf)[0]

    def integrand_t(t):
        inner_integral1 = quad(lambda u: 1 - complementary_cdf(u, distribution, *args, **kwargs), 0, t)[0]
        if inner_integral1 != 0:
            inner_integral2 = quad(lambda u: 1 - complementary_cdf(u, distribution, *args, **kwargs), t, t + 100)[0]
            return (1 - complementary_cdf(t, distribution, *args, **kwargs)) ** 2 * inner_integral1 ** (r - 1) * inner_integral2 ** (k - r)

    integral_value, _ = quad(integrand_t, 5, np.inf)

    # Round the integral value to 2 decimal places
    integral_value = round(integral_value, 2)

    equation_value = (r / expected_value_fk) * sp.comb(k, r) * integral_value
    return equation_value


def plot_results_for_all_distributions(r_values, k, distribution, distribution_params):
    results = {}
    for dist_name, dist_param in distribution_params.items():
        print(f"Calculating results for distribution: {dist_name}")
        results[dist_name] = []
        for r in r_values:
            result = calculate_equation(r, k, distribution, *dist_param)
            results[dist_name].append(result)
            print(f"r={r}, Result={result}")

        plt.plot(r_values, results[dist_name], label=dist_name)

    plt.xlabel('r')
    plt.ylabel('Result')
    plt.title(f'Results for {distribution} distribution')
    plt.legend()
    plt.grid(True)
    plt.show()

r_values = np.arange(30, 60,5)
k = 50
distribution = 'invgauss'
distribution_params = {
    'weibull': (1.5, 0, 1),
    'lognorm': (0.5, 0, 1),
    'gamma': (2, 0, 1),
    'invgauss': (1, 0, 1)
}

plot_results_for_all_distributions(r_values, k, distribution, distribution_params)
