# -*- coding: utf-8 -*-
"""
SS Astrophysics Exoplanet Assignment Q4 
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Constants & System Parameters
G = 6.6743e-11                 # Gravitational constant (m^3 kg^-1 s^-2)
M_sun = 1.25                   # Solar Masses
M_jup = 1.18                   # Jupiter Masses
M_s = M_sun*1.99e30            # Solar mass (kg)
M_j = M_jup*1.90e27            # Jupiter mass (kg)
P = 532224                     # Orbital Period (s)
e = 0.67                       # Eccentricity
w = [0, 30, 45, 90, 135, 180]      # Arguments of periastron (degrees)
i = 90                         # Inclination (degrees)

def kepler_solve(M, e, tol=1e-6):
    # Solves Kepler's equation M = E - e*sin(E) using Newton-Raphson technique
    def func(E):
        return E - e*np.sin(E) - M
    
    def func_derivative(E):
        return 1 - e*np.cos(E)
    
    E0 = M          # Initial estimate
    return newton(func, E0, fprime=func_derivative, tol=tol)

def RV_comp(t, P, e, K, w):
    # Computes the radial velocity as a function of time. 
    # Setting zero-point: t = 0 is chosen to be at periastron, meaning the planet is at its closest approach.
    M = 2*np.pi*t / P
    E = np.array([kepler_solve(Mi,e) for Mi in M])
    f = 2*np.arctan(np.sqrt((1 + e)/(1 - e))*np.tan(E/2))
    
    return K*(np.cos(f+np.radians(w)) + e*np.cos(np.radians(w)))

# Computes semi-amplitude K
K = ((2 * np.pi * G) ** (1/3) * M_j * np.sin(np.radians(i))) / (P ** (1/3) * (M_s + M_j) ** (2/3) * np.sqrt(1 - e**2))
print(f"Semi-Amplitude K: {K:.2f} m/s")

# Defines a time array for two orbital periods
time_days = np.linspace(0, 2 * P / (24 * 3600), 500)
time_seconds = time_days * 24 * 3600

# Plots RV Curve for Omega = 0
plt.figure(figsize=(10, 6))
rv_omega0 = RV_comp(time_seconds, P, e, K, 0)
plt.plot(time_days, rv_omega0, color='b')

# Find the index of the peak of the RV curve
peak_idx = np.argmax(rv_omega0)
peak_time = time_days[peak_idx]
peak_rv = rv_omega0[peak_idx]

# Plot a vertical line from the peak to the RV value of K
plt.vlines(7, peak_rv, peak_rv-K, color='r', linestyle='--', linewidth=2, label=f'K = {K:.2f} m/s')
plt.axhline(peak_rv, color='k', linestyle='--', linewidth=0.8)
plt.axhline(peak_rv-K, color='k', linestyle='--', linewidth=0.8)
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.xlim(0,12.2)
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (m/s)')
plt.title('Radial Velocity Curve for Hot Jupiter, ω = 0°', weight='bold')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# Plots RV Curve for All Omegas
plt.figure(figsize=(10, 6))
for w in w:
    rv = RV_comp(time_seconds, P, e, K, w)
    plt.plot(time_days, rv, label=f'ω = {w}°')
plt.axhline(K, color='r', linestyle='--', linewidth=1, label=f'K = {K:.2f} m/s')
plt.axhline(-K, color='r', linestyle='--', linewidth=1)
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.xlim(0,12.2)
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (m/s)')
plt.title('Radial Velocity Curve for a Hot Jupiter, with varying ω-values', weight='bold')
plt.legend()
plt.grid()
plt.show()