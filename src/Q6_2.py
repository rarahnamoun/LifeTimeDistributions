import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, pareto, weibull_min, uniform
import matplotlib.cm as cm

# Define distributions
distributions = ['exponential', 'pareto', 'weibull', 'uniform']
scale_parameter = 2  # Choose a single scale parameter value

# Define range of threshold values (v) from 0 to 1
threshold_values = np.linspace(0, 1, 100)

# Initialize TTT values for each distribution
TTT_values_dict = {distribution: [] for distribution in distributions}

# Calculate TTT diagram for each distribution and threshold value (v)
for distribution in distributions:
    if distribution == 'exponential':
        F = expon(scale=scale_parameter).cdf
        E_F = expon(scale=scale_parameter).mean()
    elif distribution == 'pareto':
        F = pareto(b=1, scale=scale_parameter).cdf
        E_F = pareto(b=1, scale=scale_parameter).mean()
    elif distribution == 'weibull':
        F = weibull_min(c=1, scale=10).cdf
        E_F = weibull_min(c=1, scale=10).mean()
    elif distribution == 'uniform':
        F = uniform(loc=0, scale=3).cdf
        E_F = uniform(loc=0, scale=3).mean()

    for v in threshold_values:
        if v == 0:
            TTT_values_dict[distribution].append(0)  # Set TTT value to zero for v=0 to avoid division by zero
        else:
            integral_value = np.trapz((1 - F(np.linspace(0, F(1 / v), 100))), dx=F(v))
            if(distribution=="exponential"):
                integral_value = np.trapz(( F(np.linspace(0, F(1 / v), 100))), dx=F(v))
            TTT = integral_value / E_F

            TTT_values_dict[distribution].append(TTT)

# Plot TTT diagrams for each distribution
plt.figure(figsize=(10, 6))
color_map = cm.get_cmap('tab10')
num_colors = len(distributions)
colors = [color_map(i) for i in np.linspace(0, 1, num_colors)]
for i, distribution in enumerate(distributions):
    plt.plot(threshold_values, TTT_values_dict[distribution], color=colors[i], linewidth=2, label=f'{distribution.capitalize()}')

plt.xlabel('Threshold (v)', fontsize=14)
plt.ylabel('TTT(v)', fontsize=14)
plt.title(f'Time-To-Threshold (TTT) Diagram for Different Distributions (Scale={scale_parameter})', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.show()
