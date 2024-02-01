# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:25:05 2023

Charlie Ashe, 21365365
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Question One and Two
# Function to calculate the Poisson probability for a given n and mean λ
def poisson_probability(n, mean):
    return (mean ** n) * math.exp(-mean) / math.factorial(n)

# Function to calculate the sum of probabilities from n=0 to N
def sum_probabilities(N, mean):
    total = 0
    for n in range(N + 1):
        total += poisson_probability(n, mean)
    return total

# Function to calculate the sum of n * P(n) from n=0 to N
def sum_n_probabilities(N, mean):
    total = 0
    for n in range(N + 1):
        total += n * poisson_probability(n, mean)
    return total

# Function to calculate the sum of n^2 * P(n) from n=0 to N
def sum_n_squared_probabilities(N, mean):
    total = 0
    for n in range(N + 1):
        total += (n ** 2) * poisson_probability(n, mean)
    return total

# Constants
N = 50

# Mean values
mean_values = [1, 5, 10]

# Table to store the results
results_table = []

# Calculating for each mean value
for mean in mean_values:
    sum_Pn = sum_probabilities(N, mean)
    sum_nPn = sum_n_probabilities(N, mean)
    sum_n2Pn = sum_n_squared_probabilities(N, mean)
    
    # Standard deviation and variance
    variance = sum_n2Pn - (sum_nPn ** 2)
    standard_deviation = math.sqrt(variance)
    
    # Verify normalization
    is_normalized = math.isclose(sum_Pn, 1.0, rel_tol=1e-9)
    
    results_table.append([mean, sum_Pn, sum_nPn, sum_n2Pn, variance, standard_deviation, is_normalized])

# Printing  results as table 
print("Mean  ∑P(n)   ∑n*P(n)   ∑n^2*P(n)   Variance   Std. Deviation  Normalized?")
for result in results_table:
    print(f"{result[0]:3}   {result[1]:.6f}   {result[2]:.6f}   {result[3]:.6f}   {result[4]:.6f}   {result[5]:.6f}   {result[6]}")

# Plotting Distributions
plt.figure(figsize=(8, 5))
for mean in mean_values:
    x = np.arange(0, 21)  # Assuming up to 20 darts for visualization
    poisson_pmf = poisson.pmf(x, mean)
    plt.plot(x, poisson_pmf,'.', label=f'Poisson (<n> = {mean})')
plt.title('Poisson Distributions for Varying Means')
plt.xlabel('Number of Darts in a Region (n)')
plt.ylabel('Probability P(n)')
plt.xlim(0.0,20.0)
plt.ylim(0,0.40)
plt.legend()
plt.grid(True)
plt.show()

# Questions 3, 4, 5, 6

# Constants
N = 50  # Number of darts thrown in one trial
L = 5 # Number of regions on the dartboard
T_values = [10, 100, 1000, 10000]  # Number of trials for simulation

# Function to perform a single dart-throwing trial and return H(n) and mean number of darts per region
def simulate_dart_trial(N, L):
    B = [0] * L  # Initialize the array to count darts in each region
    for _ in range(N):
        region = random.randint(0, L - 1)  # Choose a random region
        B[region] += 1  # Increment the count in that region
    
    H = [0] * (N + 1)  # Initialize H(n)
    mean_n = sum(B) / L  # Calculate the mean number of darts per region

    for count in B:
        H[count] += 1
    
    return H, mean_n

# Plot Psim(n) for each T value with customized y-axis ticks
x = np.arange(0, N + 1)

for T in T_values:
    H_combined = [0] * (N + 1)
    mean_n_simulation = 0
    
    for _ in range(T):
        H, mean_n = simulate_dart_trial(N, L)
        
        # Add the results to the combined data
        H_combined = [x + y for x, y in zip(H_combined, H)]
        mean_n_simulation += mean_n

    # Normalize H(n) to obtain Psim(n)
    Psim = [count / (T * L) for count in H_combined]

    # Calculate the mean mean_n
    mean_n_combined = mean_n_simulation / T

    # Customize the y-axis ticks
    y_ticks = [10**(-i) for i in range(10)]  # Customize the ticks as per your needs

    # Plot Psim(n) for each T value
    plt.figure(figsize=(8, 5))
    plt.yscale('log') 
    plt.title(f'Comparison of Psim(n) with Poisson Distribution (T = {T}, log-scale)')
    plt.xlabel('Number of Darts in a Region (n)')
    plt.ylabel('Log Probability Psim(n)')
    plt.ylim(min(y_ticks), max(y_ticks))
    plt.yticks(y_ticks, [f'{tick:.0e}' for tick in y_ticks])  
    plt.bar(x, Psim, alpha=0.6, label=f'Simulation (T = {T})')
    plt.plot(x, poisson.pmf(x, mean_n_combined), linestyle='dashed', label=f'Poisson (λ = {mean_n_combined})')

    plt.grid(True)
    plt.legend()
    plt.show()

print("Name: Charlie Ashe   Student Number: 21365365")


