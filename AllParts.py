"""
Implicit algorithms for Evolution of Quantum Wave Packets
Name: Charlie Ashe					                    
Student Number: 21365365

This code performs all five 'parts' of the simulation assignment and is organised so each part's code is as clear as possible.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import lu, solve
from scipy.integrate import simps
from scipy.ndimage import center_of_mass
from scipy.stats import linregress

size = 800   # Number of grid points
h = 1.0      # Space step
hbar = 1.0  # Reduced Planck's constant (in arbitrary units)
m = 1.0     # Mass (in arbitrary units)

#  --- Parts One & Two ---
# Define parameters
kappa1 = 0.005  # Time step 
r1 = kappa1 / h**2  # Stability parameter

def initialize_wave_function1(size):
    """Initialize the wave function as a half-sine wave."""
    x = np.arange(size)
    psi = np.sqrt(2.0) * np.sin(np.pi * x / (size - 1))
    return psi.astype(np.complex128)

def setup_hamiltonian1(size, h):
    """Set up the Hamiltonian matrix H for the finite difference method."""
    H = np.zeros((size, size), dtype=np.complex128)
    for i in range(size):
        if i > 0:
            H[i, i-1] = 1.0
        H[i, i] = -2.0
        if i < size - 1:
            H[i, i+1] = 1.0
    return -H / (2 * h**2)

def time_evolution1(psi, H, steps, kappa):
    """Evolve the wave function forward in time using the explicit method."""
    identity = np.eye(size, dtype=np.complex128)
    evolution_matrix = identity - 1j * kappa * H
    
    psi_evolution = np.zeros((steps, size), dtype=np.complex128)
    psi_evolution[0, :] = psi
    
    for t in range(1, steps):
        psi = np.dot(evolution_matrix, psi)
        psi_evolution[t, :] = psi
        
    return psi_evolution

# Initialize wave function
psi = initialize_wave_function1(size)

# Setup Hamiltonian matrix
H = setup_hamiltonian1(size, h)

# Number of time steps
steps = 1000

# Perform time evolution
psi_evolution = time_evolution1(psi, H, steps, kappa1)

# Norm Calculation
def norm_of_wave_function1(psi_evolution):
    """Evaluate the norm using Simpson's rule."""
    return np.array([simps(np.abs(psi_evolution[t, :])**2, dx=1) for t in range(psi_evolution.shape[0])])

norm = norm_of_wave_function1(psi_evolution)

# Max Time Step Detection
def find_max_time_step(kappa_values, size, H, psi, steps):
    """Find the maximum time step where norm is still conserved."""
    initial_norm = simps(np.abs(psi)**2, dx=1)
    for kappa in kappa_values:
        psi_evolution = time_evolution1(psi, H, steps, kappa)
        norm = norm_of_wave_function1(psi_evolution)
        
        # Check for norm deviation
        norm_deviation = np.abs(norm - initial_norm) / initial_norm
        if np.any(norm_deviation > 0.01):  # Deviation threshold set to 1%
            return kappa  
    return kappa_values[-1] 

# Explore different kappa values
kappa_values1 = np.linspace(0.0001, 10, 100)  # Range of possible time step values
max_kappa1 = find_max_time_step(kappa_values1, size, H, psi, steps)

print(f"Max time step before norm is no longer conserved: kappa = {max_kappa1}")

# Animation Setup
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(size)
line_real, = ax.plot([], [], label='Real Part', color='blue')
line_imag, = ax.plot([], [], label='Imaginary Part', color='red')
ax.set_xlim(0, size)
ax.set_ylim(-1.5, 2.0)  
ax.set_xlabel("Grid Points")
ax.set_ylabel("Wave Function Amplitude")
title = ax.set_title("Time Evolution of Wave Function: Step 0")  
ax.legend()

def init():
    """Initialize the animation lines."""
    line_real.set_data([], [])
    line_imag.set_data([], [])
    return line_real, line_imag, title

def update(frame):
    """Update the animation frame."""
    line_real.set_data(x, np.real(psi_evolution[frame]))
    line_imag.set_data(x, np.imag(psi_evolution[frame]))
    title.set_text(f"Time Evolution of Wave Function: Step {frame}")  # Update title with frame number
    return line_real, line_imag, title

ani = animation.FuncAnimation(fig, update, frames=steps, init_func=init, interval=30, repeat=True, blit=False)

plt.show()

#  --- Part Three ---
# Define parameters
kappa2 = 1.0  # Time step 
r2 = kappa2 / h**2  # Stability parameter

def initialize_wave_function2(size):
    """Initialize the wave function as a half-sine wave."""
    x = np.arange(size)
    psi = np.sqrt(2.0) * np.sin(np.pi * x / (size - 1))
    return psi.astype(np.complex128)

def setup_hamiltonian2(size, h):
    """Set up the Hamiltonian matrix H for the finite difference method."""
    H = np.zeros((size, size), dtype=np.complex128)
    for i in range(size):
        if i > 0:
            H[i, i-1] = 1.0
        H[i, i] = -2.0
        if i < size - 1:
            H[i, i+1] = 1.0
    return -H / (2 * h**2)

def time_evolution_implicit2(psi, H, steps, kappa):
    """Evolve the wave function forward in time using the implicit Crank-Nicholson method."""
    identity = np.eye(size, dtype=np.complex128)
    M1 = identity - 0.5j * kappa * H
    M2 = identity + 0.5j * kappa * H
    
    # LU decomposition of M1
    P, L, U = lu(M1)
    
    # Initialize array to store wave function evolution
    psi_evolution = np.zeros((steps, size), dtype=np.complex128)
    psi_evolution[0, :] = psi

    for t in range(1, steps):
        # Solve M1 * psi(t + dt) = M2 * psi(t)
        rhs = np.dot(M2, psi)
        psi_next = solve(U, solve(L, rhs))
        psi_evolution[t, :] = psi_next

        # Update psi for the next iteration
        psi = psi_next
    
    return psi_evolution

# Initialize wave function
psi = initialize_wave_function2(size)

# Setup Hamiltonian matrix
H = setup_hamiltonian2(size, h)

# Number of time steps
steps = 1000 

# Perform time evolution
psi_evolution = time_evolution_implicit2(psi, H, steps, kappa2)

# Animation
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(size)
line_real, = ax.plot([], [], label='Real Part', color='blue')
line_imag, = ax.plot([], [], label='Imaginary Part', color='red')
ax.set_xlim(0, size)
ax.set_ylim(-1.5, 2.0) 
ax.set_xlabel("Grid Points")
ax.set_ylabel("Wave Function Amplitude")
title = ax.set_title("Time Evolution of Wave Function: Step 0")  
ax.legend()

def init():
    """Initialize the animation lines."""
    line_real.set_data([], [])
    line_imag.set_data([], [])
    return line_real, line_imag, title

def update(frame):
    """Update the animation frame."""
    line_real.set_data(x, np.real(psi_evolution[frame]))
    line_imag.set_data(x, np.imag(psi_evolution[frame]))
    title.set_text(f"Time Evolution of Wave Function: Step {frame}")  # Update title with frame number
    return line_real, line_imag, title

ani = animation.FuncAnimation(fig, update, frames=steps, init_func=init, interval=30, repeat=True, blit=False)

plt.show()

# Norm Calculation 
def norm_of_wave_function2(psi_evolution):
    """Evaluate the norm using Simpson's rule."""
    return np.array([simps(np.abs(psi_evolution[t, :])**2, dx=1) for t in range(psi_evolution.shape[0])])

norm = norm_of_wave_function2(psi_evolution)

# === Max Time Step Detection ===
def find_max_time_step2(kappa_values, size, H, psi, steps):
    """Find the maximum time step where norm is still conserved."""
    initial_norm = simps(np.abs(psi)**2, dx=1)
    for kappa in kappa_values:
        psi_evolution = time_evolution_implicit2(psi, H, steps, kappa)
        norm = norm_of_wave_function2(psi_evolution)
        
        # Check for norm deviation
        norm_deviation = np.abs(norm - initial_norm) / initial_norm
        if np.any(norm_deviation > 0.01):  # Deviation threshold set to 1%
            return kappa  
    return kappa_values[-1]  

# Explore different kappa values
'''Professor-- for some reason, even though this part works perfectly and quickly when part three is separate, 
it takes forever to run here as a compilation of all parts,
so I have commented it so you can see how it was done.'''
#kappa_values2 = np.linspace(100000, 10000000, 100)
#max_kappa2 = find_max_time_step2(kappa_values2, size, H, psi, steps)

#print(f"Max time step before norm is no longer conserved for implicit: kappa = {max_kappa2}")

#  --- Part Four ---
# Define parameters
kappa3 = 100  # Time step
r = kappa3 / h**2  # Stability parameter

# Define wave number k so that one wavelength fits exactly
k_wave = 2 * np.pi / size  

def initialize_wave_function3(size):
    """Initialize the wave function as a complex plane wave centered in the grid."""
    j = np.arange(size)
    psi = np.exp(-2.0j * np.pi * (j - size / 2) / size) 
    return psi.astype(np.complex128)

def setup_hamiltonian_periodic3(size, h):
    """Set up the Hamiltonian matrix H with periodic boundary conditions."""
    H = np.zeros((size, size), dtype=np.complex128)
    for i in range(size):
        if i > 0:
            H[i, i-1] = 1.0
        H[i, i] = -2.0
        if i < size - 1:
            H[i, i+1] = 1.0

    # Add periodic boundary conditions
    H[0, size-1] = 1.0  # First connects to last
    H[size-1, 0] = 1.0  # Last connects to first

    return -H / (2 * h**2)

def time_evolution_implicit3(psi, H, steps, kappa):
    """Evolve the wave function forward in time using the implicit Crank-Nicholson method."""
    identity = np.eye(size, dtype=np.complex128)
    M1 = identity - 0.5j * kappa * H
    M2 = identity + 0.5j * kappa * H
    
    # LU decomposition of M1
    P, L, U = lu(M1)
    
    # Initialize array to store wave function evolution
    psi_evolution = np.zeros((steps, size), dtype=np.complex128)
    psi_evolution[0, :] = psi

    for t in range(1, steps):
        # Solve M1 * psi(t + dt) = M2 * psi(t)
        rhs = np.dot(M2, psi)
        psi_next = solve(U, solve(L, rhs))
        psi_evolution[t, :] = psi_next

        # Update psi for the next iteration
        psi = psi_next
    
    return psi_evolution

def norm_of_wave_function3(psi_evolution):
    """Evaluate the norm of the wave function using Simpson's rule."""
    norm = np.array([simps(np.abs(psi_evolution[t, :])**2, dx=1) for t in range(psi_evolution.shape[0])])
    return norm

# Initialize wave function
psi = initialize_wave_function3(size)

# Setup Hamiltonian matrix with periodic boundary conditions
H = setup_hamiltonian_periodic3(size, h)

# Number of time steps
steps = 1000  

# Perform time evolution using implicit Crank-Nicholson method
psi_evolution = time_evolution_implicit3(psi, H, steps, kappa3)

# Calculate norm of the wave function over time
norm = norm_of_wave_function3(psi_evolution)

# Function to find the position of the peak (maximum) and trough (minimum)
def find_peak_and_trough3(psi_evolution):
    peaks = []
    troughs = []
    for t in range(psi_evolution.shape[0]):
        # Find the position of the maximum (real part)
        peak_pos = np.argmax(np.real(psi_evolution[t, :]))
        trough_pos = np.argmin(np.real(psi_evolution[t, :])) 
        
        peaks.append(peak_pos)
        troughs.append(trough_pos)
    
    return np.array(peaks), np.array(troughs)

# Function to measure the phase velocity from the peak-trough time evolution
def measure_phase_velocity3(peaks, troughs, kappa, size):
    # Time difference for one full cycle (peak to peak)
    peak_to_peak_time = np.argmax(np.abs(peaks - peaks[0])) 
    
    # Calculate distance travelled in one cycle
    distance = size  
    
    # Time taken for the wave to move this distance
    time_taken = peak_to_peak_time * kappa * 2
    
    # Calculate phase velocity
    v_measured = distance / time_taken
    return v_measured

# Find the positions of peaks and troughs
peaks, troughs = find_peak_and_trough3(psi_evolution)

# Measure the phase velocity
v_measured = measure_phase_velocity3(peaks, troughs, kappa3, size)
print(f"Measured Phase Velocity (v_measured) = {v_measured}")

# Calculate phase velocity
v_p = hbar * k_wave / (2 * m) 
print(f"Phase Velocity (v_p) = {v_p}")

# Animation Setup
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(size)
line_real, = ax.plot([], [], label='Real Part', color='blue')
line_imag, = ax.plot([], [], label='Imaginary Part', color='red')
ax.set_xlim(0, size)
ax.set_ylim(-1.5, 2.0) 
ax.set_xlabel("Grid Points")
ax.set_ylabel("Wave Function Amplitude")
title = ax.set_title("Time Evolution of Wave Function: Step 0")  
ax.legend()

def init():
    """Initialize the animation lines."""
    line_real.set_data([], [])
    line_imag.set_data([], [])
    return line_real, line_imag, title

def update(frame):
    """Update the animation frame."""
    line_real.set_data(x, np.real(psi_evolution[frame]))
    line_imag.set_data(x, np.imag(psi_evolution[frame]))
    title.set_text(f"Time Evolution of Wave Function: Step {frame}")  # Update title with frame number
    return line_real, line_imag, title

ani = animation.FuncAnimation(fig, update, frames=steps, init_func=init, interval=30, repeat=True, blit=False)

plt.show()

# --- Part Five ---
# Parameters
size = 800  # Grid points
kappa = 100  # Time step
steps = 500  # Number of time steps
alpha = 0.005  # Wave packet width
k0 = 100 # Wave number 

def initialize_wave_packet(size, alpha, k0):
    """Initialize a Gaussian wave packet centered in the middle of the grid."""
    j = np.arange(size)
    psi = (alpha / (2.0 * np.pi))**0.25 * np.exp(-((j - size / 2) / size)**2 / (4.0 * alpha)) \
          * np.exp(1.0j * k0 * (j - size / 2))  
    return psi.astype(np.complex128)

def setup_hamiltonian_periodic(size, h):
    """Create the Hamiltonian matrix with periodic boundary conditions."""
    H = np.zeros((size, size), dtype=np.complex128)
    for i in range(size):
        if i > 0:
            H[i, i-1] = 1.0
        H[i, i] = -2.0
        if i < size - 1:
            H[i, i+1] = 1.0

    # Periodic boundary conditions
    H[0, size-1] = 1.0
    H[size-1, 0] = 1.0

    return -H / (2 * h**2)

def time_evolution_implicit(psi, H, steps, kappa):
    """Perform time evolution using the Crank-Nicholson method."""
    identity = np.eye(size, dtype=np.complex128)
    M1 = identity - 0.5j * kappa * H
    M2 = identity + 0.5j * kappa * H

    # LU decomposition
    P, L, U = lu(M1)

    psi_evolution = np.zeros((steps, size), dtype=np.complex128)
    psi_evolution[0, :] = psi

    for t in range(1, steps):
        rhs = np.dot(M2, psi)
        psi = solve(U, solve(L, rhs))
        psi_evolution[t, :] = psi

    return psi_evolution

# Initialize and evolve the wave function
psi = initialize_wave_packet(size, alpha, k0)
H = setup_hamiltonian_periodic(size, h)
psi_evolution = time_evolution_implicit(psi, H, steps, kappa)


#Measure Center of Mass and Compute Measured v_g 
com_positions = np.array([center_of_mass(np.abs(psi_evolution[t])**2)[0] for t in range(steps)])
time_steps = np.arange(steps) * kappa  # Convert to time

# Fit a linear function to center of mass motion
slope, intercept, r_value, _, _ = linregress(time_steps, com_positions)
vg_measured = slope

print(f"Measured Group Velocity (v_g) = {vg_measured:.4f}")

# Theoretical group velocity (v_g) = hbar * k0 / m
vg_theoretical = k0 
print(f"Theoretical Group Velocity (v_g) = {vg_theoretical:.4f}")

# Animation 
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(size)
line_real, = ax.plot([], [], label='Real Part', color='blue')
line_imag, = ax.plot([], [], label='Imaginary Part', color='red')
ax.set_xlim(0, size)
ax.set_ylim(-1.0, 1.0)
ax.set_xlabel("Grid Points")
ax.set_ylabel("Wave Function Amplitude")
title = ax.set_title("Wave Packet Evolution: Step 0")
ax.legend()

def init():
    """Initialize animation lines."""
    line_real.set_data([], [])
    line_imag.set_data([], [])
    return line_real, line_imag, title

def update(frame):
    """Update animation frame."""
    line_real.set_data(x, np.real(psi_evolution[frame]))
    line_imag.set_data(x, np.imag(psi_evolution[frame]))
    title.set_text(f"Wave Packet Evolution: Step {frame}")
    return line_real, line_imag, title

ani = animation.FuncAnimation(fig, update, frames=steps, init_func=init, interval=20, repeat=True, blit=False)
plt.show()
