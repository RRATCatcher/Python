# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:36:58 2023

Charlie Ashe
CompSim. A.3.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from time import time

start_time = time()

class IsingModel2D:
    def __init__(self, size, temperature, magnetic_field=0.0):
        self.size = size
        self.temperature = temperature
        self.magnetic_field = magnetic_field
        self.spins = np.ones((size, size), dtype=int)
        self.initialize_system()

    def initialize_system(self):
        self.spins = np.random.choice([-1, 1], size=(self.size, self.size))

    def energy(self):
        neighbors_sum = (
            np.roll(self.spins, 1, axis=0) +
            np.roll(self.spins, -1, axis=0) +
            np.roll(self.spins, 1, axis=1) +
            np.roll(self.spins, -1, axis=1)
        )
        return -np.sum(self.spins * neighbors_sum) - self.magnetic_field * np.sum(self.spins)

    def magnetization(self):
        return np.sum(self.spins)

    def susceptibility(self):
        return np.var(self.spins)

    def heat_capacity(self):
        return np.var(self.simulate(5000, burn_in=1000)) / (self.temperature ** 2)

    def local_field(self, x, y):
        return self.magnetic_field + (
            np.roll(self.spins, 1, axis=0)[x, y] +
            np.roll(self.spins, -1, axis=0)[x, y] +
            np.roll(self.spins, 1, axis=1)[x, y] +
            np.roll(self.spins, -1, axis=1)[x, y]
        )

    def metropolis_step(self):
        x, y = np.random.randint(0, self.size, size=2)
        dE = 2 * self.spins[x, y] * self.local_field(x, y)
        rand_vals = np.random.rand()

        if dE < 0 or rand_vals < np.exp(-dE / self.temperature):
            self.spins[x, y] *= -1

    def simulate(self, steps, burn_in=0):
        energies = []

        for _ in range(steps + burn_in):
            self.metropolis_step()

            if _ >= burn_in:
                energies.append(self.energy())

        return energies

# PART ONE: Varying h for temperatures T = 1.0, T = 4.0 

def simulate_averages1(size, temperature, magnetic_fields):
    magnetizations = []
    energies = []
    susceptibilities = []
    heat_capacities = []

    for field in magnetic_fields:
        ising_model = IsingModel2D(size, temperature, field)

        num_simulations = 20
        avg_magnetization = 0
        avg_energy = 0
        avg_susceptibility = 0
        avg_heat_capacity = 0

        for _ in range(num_simulations):
            ising_model.simulate(5000, burn_in=1000)

            avg_magnetization += ising_model.magnetization()
            avg_energy += ising_model.energy()
            avg_susceptibility += ising_model.susceptibility()
            avg_heat_capacity += ising_model.heat_capacity()

        avg_magnetization /= num_simulations
        avg_energy /= num_simulations
        avg_susceptibility /= num_simulations
        avg_heat_capacity /= num_simulations

        magnetizations.append(avg_magnetization)
        energies.append(avg_energy)
        susceptibilities.append(avg_susceptibility)
        heat_capacities.append(avg_heat_capacity)

    return magnetizations, energies, susceptibilities, heat_capacities

size = 10
magnetic_fields = np.linspace(-2.0, 2.0, 40)
temperatures = [1.0]

for temperature in temperatures:
    magnetizations, energies, susceptibilities, heat_capacities = simulate_averages1(size, temperature, magnetic_fields)

    xnew = np.linspace(min(magnetic_fields), max(magnetic_fields), 300)
    spl_magnetizations = make_interp_spline(magnetic_fields, magnetizations, k=3)
    magnetizations_smooth = spl_magnetizations(xnew)

    spl_energies = make_interp_spline(magnetic_fields, energies, k=3)
    energies_smooth = spl_energies(xnew)

    spl_susceptibilities = make_interp_spline(magnetic_fields, susceptibilities, k=3)
    susceptibilities_smooth = spl_susceptibilities(xnew)

    spl_heat_capacities = make_interp_spline(magnetic_fields, heat_capacities, k=3)
    heat_capacities_smooth = spl_heat_capacities(xnew)
    
    plt.figure(figsize=(10, 6))
    plt.plot(xnew, magnetizations_smooth,color='darkslategray', linewidth=2.5, label='Temperature: {:.2f}'.format(temperature))
    plt.xlabel('Magnetic Field', fontsize=14)
    plt.ylabel('Magnetization', fontsize=14)
    plt.xlim(-2.0, 2.0)
    plt.legend()
    plt.title('Average Magnetization vs Magnetic Field', fontsize=15, fontweight='bold')

    plt.figure(figsize=(10, 6))
    plt.plot(xnew, energies_smooth,color='darkslategray', linewidth=2.5, label='Temperature: {:.2f}'.format(temperature))
    plt.xlabel('Magnetic Field', fontsize=14)
    plt.ylabel('Energy',fontsize=14)
    plt.xlim(-2.0, 2.0)
    plt.legend()
    plt.title('Average Energy vs Magnetic Field', fontsize=15, fontweight='bold')

    plt.figure(figsize=(10, 6))
    plt.plot(xnew, susceptibilities_smooth,color='darkslategray', linewidth=2.5, label='Temperature: {:.2f}'.format(temperature))
    plt.xlabel('Magnetic Field', fontsize=14)
    plt.ylabel('Susceptibility', fontsize=14)
    plt.xlim(-2.0, 2.0)
    plt.legend()
    plt.title('Average Susceptibility vs Magnetic Field', fontsize=15, fontweight='bold')

    plt.figure(figsize=(10, 6))
    plt.plot(xnew, heat_capacities_smooth,color='darkslategray', linewidth=2.5, label='Temperature: {:.2f}'.format(temperature))
    plt.xlabel('Magnetic Field', fontsize=14)
    plt.ylabel('Heat Capacity', fontsize=14)
    plt.xlim(-2.0, 2.0)
    plt.legend()
    plt.title('Average Heat Capacity vs Magnetic Field', fontsize=15, fontweight='bold')

plt.show()

# PART TWO: Varying Temperature for h = 0 

def simulate_averages2(size, temperatures):
    magnetizations = []
    energies = []
    susceptibilities = []
    heat_capacities = []

    for temperature in temperatures:
        ising_model = IsingModel2D(size, temperature, magnetic_field=0.0)

        num_simulations = 20
        avg_magnetization = 0
        avg_energy = 0
        avg_susceptibility = 0
        avg_heat_capacity = 0

        for _ in range(num_simulations):
            ising_model.simulate(5000, burn_in=1000)

            avg_magnetization += ising_model.magnetization()
            avg_energy += ising_model.energy()
            avg_susceptibility += ising_model.susceptibility()
            avg_heat_capacity += ising_model.heat_capacity()

        avg_magnetization /= num_simulations
        avg_energy /= num_simulations
        avg_susceptibility /= num_simulations
        avg_heat_capacity /= num_simulations

        magnetizations.append(avg_magnetization)
        energies.append(avg_energy)
        susceptibilities.append(avg_susceptibility)
        heat_capacities.append(avg_heat_capacity)

    return magnetizations, energies, susceptibilities, heat_capacities

size = 10
temperatures = np.linspace(0.5, 5.0, 20)  # Change the temperature range

magnetizations, energies, susceptibilities, heat_capacities = simulate_averages2(size, temperatures)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(temperatures, magnetizations, color='darkslategray', linewidth=2.5, label='Magnetization')
plt.axvline(2.152, linestyle='--',linewidth=2.5, color='darkred', label='T_c = 2.15')
plt.xlabel('Temperature', fontsize=14)
plt.ylabel('Magnetization', fontsize=14)
plt.xlim(0.5,5)
plt.legend()
plt.title('Average Magnetization vs Temperature', fontsize=15, fontweight='bold')

plt.figure(figsize=(10, 6))
plt.plot(temperatures, energies, color='darkslategray', linewidth=2.5, label='Energy')
plt.xlabel('Temperature', fontsize=14)
plt.ylabel('Energy', fontsize=14)
plt.xlim(0.5,5)
plt.legend()
plt.title('Average Energy vs Temperature', fontsize=15, fontweight='bold')

plt.figure(figsize=(10, 6))
plt.plot(temperatures, susceptibilities, color='darkslategray', linewidth=2.5, label='Susceptibility')
plt.xlabel('Temperature', fontsize=14)
plt.ylabel('Susceptibility', fontsize=14)
plt.xlim(0.5,5)
plt.legend()
plt.title('Average Susceptibility vs Temperature', fontsize=15, fontweight='bold')

plt.figure(figsize=(10, 6))
plt.plot(temperatures, heat_capacities, color='darkslategray', linewidth=2.5, label='Heat Capacity')
plt.xlabel('Temperature', fontsize=14)
plt.ylabel('Heat Capacity', fontsize=14)
plt.xlim(0.5,5)
plt.legend()
plt.title('Average Heat Capacity vs Temperature', fontsize=15, fontweight='bold')

plt.show()

# PART THREE: At Critical Temperature Tc = 2.15

def simulate_averages3(size_values, temperature):
    susceptibilities = []
    heat_capacities = []

    for size in size_values:
        ising_model = IsingModel2D(size, temperature)

        num_simulations = 20
        avg_susceptibility = 0
        avg_heat_capacity = 0

        for _ in range(num_simulations):
            ising_model.simulate(5000, burn_in=1000)

            avg_susceptibility += ising_model.susceptibility()
            avg_heat_capacity += ising_model.heat_capacity()

        avg_susceptibility /= num_simulations
        avg_heat_capacity /= num_simulations

        susceptibilities.append(avg_susceptibility)
        heat_capacities.append(avg_heat_capacity)

    return susceptibilities, heat_capacities

# Set the temperature
temperature = 2.15

# Set the range of system sizes
size_values = (2,12,1)

susceptibilities, heat_capacities = simulate_averages3(size_values, temperature)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(size_values, susceptibilities, marker='o', linestyle='-', color='darkslategray', linewidth=2.5, label='Susceptibility')
plt.xlabel('System Size', fontsize=14)
plt.ylabel('Magnetic Susceptibility', fontsize=14)
plt.legend()
plt.title('Average Magnetic Susceptibility vs System Size at T = 2.15', fontsize=15, fontweight='bold')

plt.figure(figsize=(10, 6))
plt.plot(size_values, heat_capacities, marker='o', linestyle='-', color='darkslategray', linewidth=2.5, label='Heat Capacity')
plt.xlabel('System Size', fontsize=14)
plt.ylabel('Heat Capacity', fontsize=14)
plt.legend()
plt.title('Average Heat Capacity vs System Size at T = 2.15', fontsize=15, fontweight='bold')

plt.show()

end_time = time()

print('Total time =', end_time - start_time, 's')

