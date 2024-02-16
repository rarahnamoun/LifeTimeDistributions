import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, pareto

# Parameters for the distributions
lambda_param_exp = 0.25  # Rate parameter for exponential distribution
shape_param_pareto = 0.15  # Shape parameter for Pareto distribution

# Generate data points for the distributions
x = np.linspace(1, 20, 1000)

# Calculate the PDFs of the distributions
pdf_exponential = expon.pdf(x, scale=1/lambda_param_exp)
pdf_pareto = pareto.pdf(x, b=shape_param_pareto)

# Plotting the PDFs
plt.figure(figsize=(10, 6))

plt.plot(x, pdf_exponential, label='Exponential Distribution', color='blue')
plt.plot(x, pdf_pareto, label='Pareto Distribution', color='red')

plt.title('Probability Density Functions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

plt.show()
