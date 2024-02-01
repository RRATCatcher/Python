#Radiometer Equation Code for SETI Internship, 24.07.2023, Charlie K. Ashe

"""
This code can be used to calculate the flux density [S] of some source of radiation given its observed frequency. 
"""

def flux_density_calc(T, A, B, t, N, P):
    # Boltzmann constant [Multipled by 1e26 for Jansky conversion]
    k = 1380

    # Flux Density Calculation
    S = (2 * k * T * N) / (A * (B * t * P)**0.5)

    return S

# Inputs:
system_temperature = 20.0               # System Temperature in Kelvin
effective_area = 70000                   # Effective Area in square metres
bandwidth = 195312.5                # Bandwidth of Observation in Hz
integration_time =  9343.0                # Integration time in seconds
number_of_polarisations = 2.0       # Number of polarisations [typically 2]
noise = 7.0                            # Noise for signal to noise ratio

#The example numbers above are the most accurate numbers I could find within the FAST paper about J1913+1330

flux_density = flux_density_calc(system_temperature,effective_area,bandwidth,integration_time,noise,number_of_polarisations)
print("Flux Density =", flux_density, "Jy")
