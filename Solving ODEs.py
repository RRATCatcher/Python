"""
Computer Simulations I Assignment Three 
Charlie Ashe, 21365365
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the ODE function f(x, t)
def f(x, t):
    return ((1 + t) * x) + 1 - (3 * t) + t**2

# Create a grid of t and x values
t = np.linspace(0, 5, 25)
x = np.linspace(-3, 3, 25)
T, X = np.meshgrid(t, x)

# Calculating dx/dt at each grid point using the ODE function
dxdt = f(X, T)

# Creating the direction field plot
plt.figure(figsize=(10, 6))

# Setting variables
step_size = 0.02
x0_euler = 0.0655
x0_improved_euler = 0.0655
x0_rk4 = 0.0655

# Lists to store values for each method 
t_values = [0]
x_values_euler = [x0_euler]
x_values_improved_euler = [x0_improved_euler]
x_values_rk4 = [x0_rk4]

# Euler's method, Improved Euler, and Runge-Kutta methods
for t in np.arange(0, 5, step_size):
    # Euler's method
    x0_euler = x0_euler + step_size * f(x0_euler, t)
    x_values_euler.append(x0_euler)

    # Improved Euler's method
    k1 = f(x0_improved_euler, t)
    k2 = f(x0_improved_euler + step_size * k1, t + step_size)
    x0_improved_euler = x0_improved_euler + (step_size / 2) * (k1 + k2)
    x_values_improved_euler.append(x0_improved_euler)

    # Runge-Kutta method 
    k1 = f(x0_rk4, t)
    k2 = f(x0_rk4 + (step_size / 2) * k1, t + step_size / 2)
    k3 = f(x0_rk4 + (step_size / 2) * k2, t + step_size / 2)
    k4 = f(x0_rk4 + step_size * k3, t + step_size)
    x0_rk4 = x0_rk4 + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    x_values_rk4.append(x0_rk4)

    t_values.append(t + step_size)

# Plotting the numerical solutions for all methods
plt.plot(t_values, x_values_euler, label='Euler Method', color='red')
plt.plot(t_values, x_values_improved_euler, label='Improved Euler (Heun\'s Method)', color='blue')
plt.plot(t_values, x_values_rk4, label='Runge-Kutta (4th Order)', color='green')

plt.quiver(T, X, np.ones_like(dxdt), dxdt, scale=30, color='gray', alpha=0.5, label='Direction Field')

plt.xlabel('t')
plt.ylabel('x')
plt.xlim(0,4)
plt.ylim(-10, 10)
plt.title('Comparison of Numerical Methods to solve an ODE')
#plt.title('Solution to ODE using Euler Method and stepsize = 0.04')
plt.legend()
plt.grid(True)
plt.show()


print("Name: Charlie Ashe     Student Number: 21365365")
