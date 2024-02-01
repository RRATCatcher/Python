# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:50:02 2023

Charlie Ashe
CompSci A.2.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

u = 1  # in eV
rho_fixed = 2.27e27  # in nm^(-3)
T_range = np.linspace(0, 10e4, 100)
kB = 8.62e-5  # in eV/K
h_bar = 0.276  # in nm * √eV*me.
me = 9.10938356e-31  # electron mass in kg
pi = np.pi

# Function for Fermi-Dirac distribution
def f(w, T, u):
    return 1 / (np.exp((w - u) / (kB * T)) + 1)

# Function for density of states
def d(w):
    return (2 * me**(1.5) * w**(0.5)) / (np.pi**2 * h_bar**3)

# Function for the integrand in the expression for energy density
def integrand(w, T, u):
    return w * d(w) * f(w, T, u)

# Function to calculate energy density at fixed chemical potential
def energy_density_fixed_chemical_potential(T):
    result, _ = integrate.quad(lambda w: integrand(w, T, u), 0, np.inf)
    return result

# Calculate energy density for each temperature in the range
energy_density_fixed_chemical_potential_values = [energy_density_fixed_chemical_potential(temp) for temp in T_range]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(T_range, energy_density_fixed_chemical_potential_values,linewidth=2.0,color='darkblue', label='Energy Density at Fixed Chemical Potential')
plt.xlabel('Temperature (K)',fontsize=12)
plt.ylabel('Energy Density',fontsize=12)
plt.xlim(0,)
plt.ylim(0,)
plt.title('Energy Density of 3D Electron Gas',fontsize=15,fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()

# Question Two 

def AAH_Hamiltonian(alpha, L, u, h_bar, tun):
    # Define the on-site energy
    epsilon = u * np.cos(2 * np.pi * alpha * np.arange(L))
    
    # Create the Hamiltonian matrix
    H = np.diag(epsilon) + np.diag(-h_bar * tun * np.ones(L-1), k=1) + np.diag(-h_bar * tun * np.ones(L-1), k=-1)
    
    return H

def compute_and_plot_eigenvalues(alpha, L, u, h_bar, tun, color):
    # Define the Hamiltonian
    H = AAH_Hamiltonian(alpha, L, u, h_bar, tun)
    
    # Find the eigenvalues
    w = np.linalg.eigvals(H)
    
    # Sort the eigenvalues
    w.sort()

    # Plot the eigenvalues with the specified colour and label
    plt.plot(w, '.', label=f'α = {alpha}', color=color)

def compute_and_plot_DOS(alpha, L, u, h_bar, tun, delta_w, color):
    # Define the Hamiltonian
    H = AAH_Hamiltonian(alpha, L, u, h_bar, tun)
    
    # Find the eigenvalues
    w = np.linalg.eigvals(H)
    
    # Compute the density of states (DOS)
    dos, bins = np.histogram(w, bins=np.arange(min(w), max(w)+delta_w, delta_w))
    
    # Plot the density of states with the specified colour and label
    plt.bar(bins[:-1], dos, width=delta_w, alpha=0.7, label=f'α = {alpha}', color=color)

# Parameters
L = 500
u = 2.0
h_bar = 1.0
tun = 1.0
delta_w = h_bar * tun / 10

# List of alpha values with corresponding colours
alpha_color_mapping = [(1.0, 'sandybrown'), (0.5, 'lightseagreen'), (1/3, 'darkblue')]

plt.figure(figsize=(10, 6))
for alpha, color in alpha_color_mapping:
    compute_and_plot_eigenvalues(alpha, L, u, h_bar, tun, color)

plt.title('Eigenvalues of the AAH Model with Varying Alphas', fontsize=15, fontweight='bold')
plt.xlabel('Eigenvalue Order', fontsize=14)
plt.ylabel('Eigenvalues', fontsize=14)
plt.xlim(-20, 520)
plt.ylim(-3, 5)
plt.legend()
plt.show()

# Plot density of states for each alpha with colours
plt.figure(figsize=(10, 6))
for alpha, color in alpha_color_mapping:
    compute_and_plot_DOS(alpha, L, u, h_bar, tun, delta_w, color)

plt.title('Density of States of the AAH Model', fontsize=15, fontweight='bold')
plt.xlabel('Eigenvalues', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xlim(-3, 4)
plt.ylim(0, 65)
plt.legend()
plt.show()

def compute_mean_IPR(alpha, L, u_values, h_bar, tun):
    mean_ipr_values = []

    for u in u_values:
        H = AAH_Hamiltonian(alpha, L, u, h_bar, tun)
        w, v = np.linalg.eigh(H)
        ipr_values = np.sum(v**4, axis=0) / np.sum(v**2, axis=0)**2
        mean_ipr = np.mean(ipr_values)
        mean_ipr_values.append(mean_ipr)

    plt.plot(u_values, mean_ipr_values,color='darkblue', linewidth=3, label=f'α = {alpha}')

# Parameters
L = 500
h_bar = 1.0
tun = 1.0
u_values = np.linspace(0.0, 5.0, 50)

# Golden ratio alpha
golden_ratio_alpha = (1 + np.sqrt(5)) / 2

# Plot mean IPR for the golden ratio alpha
plt.figure(figsize=(10, 6))
compute_mean_IPR(golden_ratio_alpha, L, u_values, h_bar, tun)

plt.title('Mean IPR for the Golden Ratio Alpha', fontsize=15,fontweight='bold')
plt.xlabel('$u/\\bar{h}\\lambda$',fontsize=14)
plt.ylabel('Mean IPR',fontsize=14)
plt.xlim(0,)
plt.ylim(0,)
plt.axvline(1.93, linestyle='--',linewidth=2.5, color='darkred', label='u_c = 1.93')
plt.legend()
plt.show()

# Critical disorder strength
critical_disorder_strength = 1.93

# Choose eigenstates for visualization
extended_state_index = 0  # Index of an eigenstate in the extended phase
localized_state_index = -1  # Index of an eigenstate in the localized phase

# Compute Hamiltonian for extended and localized states
H_extended = AAH_Hamiltonian(golden_ratio_alpha, L, critical_disorder_strength - 1.0, h_bar, tun)
H_localized = AAH_Hamiltonian(golden_ratio_alpha, L, critical_disorder_strength + 1.0, h_bar, tun)

# Diagonalize the Hamiltonian to get eigenstates
w_extended, v_extended = np.linalg.eigh(H_extended)
w_localized, v_localized = np.linalg.eigh(H_localized)

# Plot spatial probability distribution for extended state
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.abs(v_extended[:, extended_state_index])**2,linewidth=2, color='darkblue', label=f'Extended State (u < uc)')
plt.title('Spatial Probability Distribution for an Extended State', fontsize=13,fontweight='bold')
plt.xlim(0,)
plt.ylim(0,)
plt.xlabel('Lattice Site Index',fontsize=12)
plt.ylabel('Probability Density',fontsize=12)
plt.legend()

# Plot spatial probability distribution for localized state
plt.subplot(1, 2, 2)
plt.plot(np.abs(v_localized[:, localized_state_index])**2, linewidth=2, color='darkred', label=f'Localized State (u > uc)')
plt.title('Spatial Probability Distribution for a Localized State', fontsize=13,fontweight='bold')
plt.xlim(0,)
plt.ylim(0,1.0)
plt.xlabel('Lattice Site Index',fontsize=12)
plt.ylabel('Probability Density',fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# Question Three 

# Function to create the tight-binding Hamiltonian matrix with impurity
def tight_binding_impurity(L, u, v, l0):
    H = np.zeros((L, L), dtype=complex)
    for l in range(L):
        for m in range(L):
            if l == m:
                H[l, m] = u if l != l0 else v
            elif abs(l - m) == 1:
                H[l, m] = -1  # Set off-diagonal elements to -1 (as in the question)
    return H

# Define the parameters
L = 500
tun = 1.0
u = 2.0 
v = 5.0 
l_0 = 249  # Impurity location

# Define the Hamiltonian
H = tight_binding_impurity(L, u, v, l_0)

# Define the initial state |ψ(0)⟩
psi_0 = np.zeros(L, dtype=complex)
psi_0[98] = 1.0 / np.sqrt(2)
psi_0[99] = 1.0j / np.sqrt(2)

# Solve the differential equation and plot results
fig, ax = plt.subplots(4, 1, figsize=(8, 8))
ax[0].plot(np.abs(psi_0)**2,color='darkslategray')
ax[0].set_title('$\\lambda t = 0$',fontsize=15)
ax[0].set_ylabel('$|\\psi_l|^2$', fontsize=15) 
ax[0].set_xlabel('$l$', fontsize=15)  
ax[0].axvline(249, linestyle='--',linewidth=2.5, color='darkred', label='Impurity l0 = 249')
ax[0].legend()

ts = [10, 50, 100]
for jj in range(len(ts)):
    t_f = ts[jj]
    sol = integrate.solve_ivp(lambda t, y: -1j * H @ y, (0.0, t_f), psi_0)
    ax[jj+1].plot(np.abs(sol.y[:,-1])**2, color='darkslategray')
    ax[jj+1].set_title('$\\lambda t =$ ' + str(t_f), fontsize=15)  
    ax[jj+1].set_ylabel('$|\\psi_l|^2$', fontsize=15) 
    ax[jj+1].set_xlabel('$l$', fontsize=15)
    ax[jj+1].axvline(249, linestyle='--',linewidth=2.5, color='darkred', label='Impurity l0 = 249')
    ax[jj+1].legend()
    
plt.suptitle('Spatial Probability Distribution at Different Points in Time', fontsize=16,fontweight='bold')
fig.tight_layout()
plt.show()


