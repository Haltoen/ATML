import numpy as np
import matplotlib.pyplot as plt
import math
n_reps = 1000000 # num samples
n_draws = 20 # draws pet sample
bias = 0.5 # Bernoulli bias
alpha_vals = np.arange(0.5, 1.05, 0.05) # Possible outcomes

#samples and proccesing
Samples =  np.random.binomial(n_draws, bias, n_reps)/n_draws
frequencies = np.mean(Samples[:, np.newaxis] >= alpha_vals, axis=0)

Markov_bound = bias / alpha_vals # E(x_i)=0.5
var = bias * (1- bias)

Chebyshev_bound = np.ones_like(alpha_vals)  # Initialize with 1
Hoeffding_bound = np.ones_like(alpha_vals)  # Initialize with 1

for i, alpha in enumerate(alpha_vals):
    if alpha > 0.5:
        epsilon = alpha - 0.5  # Deviation from the mean
        Chebyshev_bound[i] = min(1, var/((epsilon**2)*n_draws))  # Apply Chebyshev's inequality
        Hoeffding_bound[i] = min(1, math.exp(-2*n_draws*(epsilon**2)))



        #Corollary 2.5.



plt.figure()
plt.plot(alpha_vals, frequencies, marker='o', linestyle='-', color='b', label='Empirical Frequency')
plt.plot(alpha_vals, Markov_bound, marker='o', linestyle='-', color='r', label="Markov's Bound")
plt.plot(alpha_vals, Chebyshev_bound, marker='o', linestyle='-', color='g', label="Chebyshev's Bound")
plt.plot(alpha_vals, Hoeffding_bound, marker='o', linestyle='-', color='y', label="Hoeffding's Bound")
plt.legend()
plt.title("Bounds for α in [0.5, 1]")
plt.xlabel("Alpha (α)")
plt.ylabel("Empirical Frequency")
plt.grid(True)
plt.show()


bias = 0.1 # Bernoulli bias

#samples and proccesing
Samples =  np.random.binomial(n_draws, bias, n_reps)/n_draws
frequencies = np.mean(Samples[:, np.newaxis] >= alpha_vals, axis=0)

Markov_bound = bias / alpha_vals # E(x_i)=0.1
var = bias * (1- bias)

Chebyshev_bound = np.ones_like(alpha_vals)  # Initialize with 1
Hoeffding_bound = np.ones_like(alpha_vals)  # Initialize with 1

for i, alpha in enumerate(alpha_vals):
    if alpha > 0.5:
        epsilon = alpha - 0.5  # Deviation from the mean
        Chebyshev_bound[i] = min(1, var/((epsilon**2)*n_draws))  # Apply Chebyshev's inequality
        Hoeffding_bound[i] = min(1, math.exp(-2*n_draws*(epsilon**2)))



        #Corollary 2.5.



plt.figure()
plt.plot(alpha_vals, frequencies, marker='o', linestyle='-', color='b', label='Empirical Frequency')
plt.plot(alpha_vals, Markov_bound, marker='o', linestyle='-', color='r', label="Markov's Bound")
plt.plot(alpha_vals, Chebyshev_bound, marker='o', linestyle='-', color='g', label="Chebyshev's Bound")
plt.plot(alpha_vals, Hoeffding_bound, marker='o', linestyle='-', color='y', label="Hoeffding's Bound")
plt.legend()
plt.title("Bounds for α in [0.5, 1]")
plt.xlabel("Alpha (α)")
plt.ylabel("Empirical Frequency")
plt.grid(True)
plt.show()