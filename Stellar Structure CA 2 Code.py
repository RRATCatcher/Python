# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 20:40:05 2023

@author: charl
"""
import numpy as np
import matplotlib.pyplot as plt

# Constants 
eV_2_J = 1.602e-19    # To convert eV to J
Me = 9.109e-31        # Mass of Electron in kg
ne = 65e19          # Number density of free electrons in m^-3 using the last two digits of student number, 65
h = 6.626e-34       # Plank's constant in J*s
kB = 1.381e-23       # Boltzmann constant in J/K
ionization_energy_CaI = 6.11  # Ionization energy of Ca I in eV

# Temperature range
T = np.linspace(2000, 22000, 500)

# Partition functions for Ca I and Ca II
Z_I = 1.32
Z_II = 2.30

# Ionisation fraction function for Ca II
def ion_func_CaII(T):
    prefactor = 2 / (ne * Z_II)
    maxwell_boltzmann = ((2 * np.pi * Me * kB * T) / h**2)**(3/2) * np.exp(-ionization_energy_CaI * eV_2_J / (kB * T))
    ion_frac_CaII = prefactor * maxwell_boltzmann / (1 + prefactor * maxwell_boltzmann)
    return ion_frac_CaII

# Plotting
plt.figure(figsize=(8,5))
plt.plot(T, ion_func_CaII(T), marker='.', color='orange', linestyle='-', label='Fraction of Ca II', markersize=6)
plt.xlabel('Temperature (T)')
plt.ylabel('Ionisation Fraction (Ca II / Ca I)')
plt.title('Ionisation Fraction of Ca II in Stellar Atmosphere')
plt.grid(True)
plt.xlim(2000,22200)
plt.ylim(0,)
plt.legend()
plt.show()

# Partition function [Three Energy Levels]
def part_func(T):
    Z = 2 + 8*np.exp((-10.2*eV_2_J)/(kB*T)) + 18*np.exp((-12.1*eV_2_J)/(kB*T))
    return Z

# Ionisation fraction function
def ion_func(T):
    Z = part_func(T)
    prefactor = 2 / (ne * Z)
    maxwell_boltzmann = ((2 * np.pi * Me * kB * T) / h**2)**(3/2) * np.exp(-10.2 * eV_2_J / (kB * T))
    ion_frac = prefactor * maxwell_boltzmann / (1 + prefactor * maxwell_boltzmann)
    return ion_frac

# Plotting
plt.figure(figsize=(8,5))
plt.plot(T, ion_func(T), marker='.', color='red', label='Ionisation Fraction of Hydrogen')
plt.xlabel('Temperature (T)')
plt.ylabel('Ionisation Fraction (HII/HI)')
plt.title('Ionisation Fraction of H in Stellar Atmosphere')
plt.xlim(2000,22200)
plt.ylim(0,)
plt.legend()
plt.grid(True)
plt.show()

# Iterating through temperatures to find when ionization fraction is 0.9
for temp, ion_func in zip(T, ion_func(T)):
    if ion_func >= 0.9:
        estimated_temperature = temp
        break

print(f"Estimated temperature at 90% ionization: {estimated_temperature:.2f} K")

