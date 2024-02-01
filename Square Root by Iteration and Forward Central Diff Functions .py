# -*- coding: utf-8 -*-
"""
Computer Simulations I Assignment One 
Charlie Ashe, 21365365
"""
# SECTION A 

import matplotlib.pyplot as plt
import numpy as np 

# Defining function for finding the square root where numit is the number of iterations
def iteration_sqrt(a, x0, numit):
    x_values = []   
    r_errors = []
    
    xn = x0
    for n in range(numit) :
        x_values.append(xn)
        xn_1 = 0.5 * (xn + (a / xn))
        xn = xn_1
        r_error = abs(xn_1*xn_1 - a) / a
        r_errors.append(r_error)
        xn = xn_1
    return x_values, r_errors

# Compute square root
a = 2
numit = 20
x0_values = [1.0, 1.5, 2.0]

for x0 in x0_values:
    x_values, r_errors = iteration_sqrt(a, x0, numit)  # Separate x_values and r_errors
    
    # Print the approximate square root
    approximate_square_root = x_values[-1]
    print(f"Approximate square root of {a} with x0 = {x0}: {approximate_square_root}")

# Relative Error

for x0 in x0_values:
    plt.figure(figsize=(6, 4))  # Create a new figure for each set of plots
    x_values, r_errors = iteration_sqrt(a, x0, numit)
    
    # Plot xn
    plt.plot(range(numit), x_values, 'o')
    plt.xlabel('Iteration')
    plt.ylabel('xn')
    plt.axhline(y=1.41421356237, color='r', linestyle='--', label='sqrt(2)')
    plt.title(f'Square Root Approximation (x0 = {x0})')
    plt.xlim([0, numit - 1])
    plt.ylim([0.8, 2.2])
    plt.legend()
    
    plt.tight_layout()

    # Create a new figure for relative error
    plt.figure(figsize=(6, 4))
    plt.plot(range(numit), r_errors, 'o')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Error')
    plt.title(f'Relative Error in xn * xn (x0 = {x0})')
    plt.xlim([0, numit - 0.8])
    plt.ylim([0, max(r_errors)+0.005])
    plt.legend()
    
    plt.tight_layout()

plt.show()

# SECTION B 

# Function to find the range of numbers valid within Python
def minmax(small,big,numit):
    for n in range(numit):
        small = small / 2
        big = big * 2
        print(n, ";", small, ";", big)
        
    return n

small = 1.0
big =  1.0
numit = 1080
minmax(small, big, numit)

# Function to find machine precision 
def machine_prec(eps,numit):
    for n in range(numit):
        eps = eps/2
        one = 1 + 1j + eps
        print(n,";", one)
    return n

eps = 1
numit = 53
machine_prec(eps,numit)

# Defining functions
def cost(t):
    return np.cos(t)

def exp_t(t):
    return np.exp(t)

# True derivatives
def true_derivative_cost(t):
    return -np.sin(t)

def true_derivative_exp_t(t):
    return np.exp(t)

# Defining the Forward Diff. and Central Diff. functions
def forward_difference(func, t, h):
    return (func(t + h) - func(t)) / h

def central_difference(func, t, h):
    return (func(t + h/2) - func(t - h/2)) / h

# Values of t 
t_values = [0.1, 1.0, 100.0]

# Range of h 
h_values = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 2e-16]
max_h = max(h_values)
min_h = min(h_values)

# Arrays to store results
forward_errors_cost = []
forward_errors_exp_t = []
central_errors_cost = []
central_errors_exp_t = []

# Calculating derivatives and relative errors for each combination of t and h
for t in t_values:
    forward_errors_cost_t = []
    forward_errors_exp_t_t = []
    central_errors_cost_t = []
    central_errors_exp_t_t = []
    
    true_derivative_cost_t = true_derivative_cost(t)
    true_derivative_exp_t_t = true_derivative_exp_t(t)
    
    for h in h_values:
        forward_diff_cost = forward_difference(cost, t, h)
        forward_diff_exp_t = forward_difference(exp_t, t, h)
        central_diff_cost = central_difference(cost, t, h)
        central_diff_exp_t = central_difference(exp_t, t, h)
        
        forward_error_cost = abs(forward_diff_cost - true_derivative_cost_t) / abs(true_derivative_cost_t)
        forward_error_exp_t = abs(forward_diff_exp_t - true_derivative_exp_t_t) / abs(true_derivative_exp_t_t)
        central_error_cost = abs(central_diff_cost - true_derivative_cost_t) / abs(true_derivative_cost_t)
        central_error_exp_t = abs(central_diff_exp_t - true_derivative_exp_t_t) / abs(true_derivative_exp_t_t)
        
        forward_errors_cost_t.append(forward_error_cost)
        forward_errors_exp_t_t.append(forward_error_exp_t)
        central_errors_cost_t.append(central_error_cost)
        central_errors_exp_t_t.append(central_error_exp_t)
    
    # Plots
    plt.figure(figsize=(10, 6))
    
    plt.loglog(h_values, forward_errors_cost_t, 'o-', label='Forward Difference [cos(t)]')
    plt.loglog(h_values, central_errors_cost_t, 'o-', label='Central Difference [cos(t)]')
    plt.loglog(h_values, forward_errors_exp_t_t, 'o-', label='Forward Difference [exp(t)]')
    plt.loglog(h_values, central_errors_exp_t_t, 'o-', label='Central Difference [exp(t)]')
    plt.title(f'Relative Error vs h for Derivates Evaluated at t = {t}')
    plt.xlabel('log10(h)')
    plt.ylabel('log10(Relative Error)')
    plt.xlim(10e-1, 10e-17)
    plt.legend()
    
# Show plots
plt.show()

print("Name: Charlie Ashe     Student Number: 21365365")
