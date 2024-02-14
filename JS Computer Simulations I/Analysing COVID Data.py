# -*- coding: utf-8 -*-
"""
Computer Simulations I Assignment Two 
Charlie Ashe, 21365365
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Opening data
days, cases = np.loadtxt("COVIDData.dat", skiprows=1, unpack=True)

# Recreating the plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(days, cases, '.',color='black', label="Daily Cases")
plt.xlabel("Days")
plt.ylabel("Daily Cases")
plt.title("COVID-19 Daily Cases in the Republic of Ireland")
plt.xlim(0,600)
plt.ylim(0,8500)
plt.grid(True)
plt.legend()
plt.show()

# Take the natural logarithm of daily cases
ln_cases = np.log(cases)

# Making log plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(days, ln_cases,'.',color='black', label="ln(Daily Cases)")
plt.xlabel("Days")
plt.ylabel("ln(Daily Cases)")
plt.title("ln(Daily Cases) Over Time")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()

# Take the natural logarithm of daily cases
ln_cases = np.log(cases)

# Function for the linear fit (Eqn. 4: ln n(t) = a + bt)
def linear_fit(t, a, b):
    return a + b * t

# Function to fit a linear segment
def fit_linear_segment(t, y, t0, fit_range):
    # Defining the indices within the specified fit_range
    fit_mask = (t >= t0) & (t < (t0 + fit_range))
    
    # Performing the linear fit using curve_fit
    params, _ = curve_fit(linear_fit, t[fit_mask], y[fit_mask])
    
    return params  # Return the parameters a and b

# Define the segments and fit ranges
segments = [
    {"t0": 0, "fit_range": 30},  # Wave One; Growth
    {"t0": 60, "fit_range": 60},  # Wave One; Decay
    {"t0": 120, "fit_range": 110},  # Wave Two; Growth
    {"t0": 235, "fit_range": 40},  # Wave Two; Decay
    {"t0": 280, "fit_range": 35},  # Wave Three; Growth
    {"t0": 315, "fit_range": 45},  # Wave Three; Decay
    
]

# Perform linear fits for each segment
fit_parameters = []
for segment in segments:
    t0 = segment["t0"]
    fit_range = segment["fit_range"]
    params = fit_linear_segment(days, ln_cases, t0, fit_range)
    fit_parameters.append(params)

# The 'fit_parameters' list contains the 'a' and 'b' values for each segment
# The 'segments' list contains the 't0' values for each segment

table_data = []

for i, segment in enumerate(segments):
    t0 = segment["t0"]
    a, b = fit_parameters[i]
    
    # Calculate n0 and λ using the equations
    n0 = np.exp(a + b * t0)
    lambda_val = b
    
    table_data.append([t0, a, b, n0, lambda_val])

# Display table
print("t0\t   a\t       b\t      n0\t         λ")
for row in table_data:
    t0, a, b, n0, lambda_val = row
    print(f"{t0}\t {a:.4f}\t {b:.4f}\t {n0:.4f}\t {lambda_val:.4f}")


# Displaying the fit parameters for each segment
for i, params in enumerate(fit_parameters):
    a, b = params
    print(f"Segment {i + 1}: a = {a}, b = {b}")

# Creating a plot to visualize the fitted segments
plt.figure(figsize=(10, 6))
plt.plot(days, ln_cases,'.', color='black', label="ln(Daily Cases)")
for segment in segments:
    t0 = segment["t0"]
    fit_range = segment["fit_range"]
    fit_mask = (days >= t0) & (days < (t0 + fit_range))
    plt.plot(days[fit_mask], linear_fit(days[fit_mask], *fit_linear_segment(days, ln_cases, t0, fit_range)),linewidth=4.0, label=f"Segment {t0}-{t0+fit_range}")
plt.xlabel("Days")
plt.ylabel("ln(Daily Cases)")
plt.xlim(0,)
plt.ylim(0,)
plt.title("Log of COVID-19 Daily Cases in the Republic of Ireland with Linear Regression")
plt.grid(True)
plt.legend()
plt.show()


# Define the function for the exponential curve
def exponential_curve(t, n0, lambda_val, t0):
    return n0 * np.exp(lambda_val * (t - t0))

# List of values for n0 and λ for each segment
n0_values = [1.2716, 325.9574,9.4469,830.6105,159.7284,3521.7150]
lambda_values = [0.2162,-0.0684,0.0413,-0.0326,0.1071,-0.0431]
t0_values = [0.0,60.0,120.0,235.0,280.0,315.0]

# Create a plot to visualize the COVID data and the exponential curves
plt.figure(figsize=(10, 6))
plt.plot(days, cases, '.', label="COVID Data", color='black')
for n0, lambda_val, t0 in zip(n0_values, lambda_values, t0_values):
    curve = exponential_curve(days, n0, lambda_val, t0)
    plt.plot(days, curve,linewidth=3.0, label=f"n0={n0:.2f}, λ={lambda_val:.4f}, t0={t0}")
plt.xlabel("Days")
plt.ylabel("Number of Cases")
plt.title("COVID Data with Exponential Curves")
plt.xlim(0,)
plt.ylim(0,9000)
plt.grid(True)
plt.legend()
plt.show()

print("Name: Charlie Ashe     Student Number: 21365365")
