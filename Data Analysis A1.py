# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy 
from astropy.io import ascii

    #**DATASET A**
# Using astropy.io to read Dataset A
data_A = ascii.read('Dataset A.txt')

print(data_A)

# Calculate the mean and standard deviation of y values in Dataset A
mean_y_A = np.mean(data_A['y'])
std_dev_y_A = np.std(data_A['y'])

# Print the calculated mean and standard deviation
print("Mean of y in Dataset A:", mean_y_A)
print("Standard Deviation of y in Dataset A:", std_dev_y_A)

# This creates a linear least-squares regression for x and y columns in data_A
llsrA = scipy.stats.linregress(data_A['x'], data_A['y'])

print(llsrA)

# Making a plot with error bars for Data_A

plt.figure(figsize=(7,4))
plt.plot(data_A['x'], data_A['y'], '.', label='Data A', color='black')
plt.errorbar(data_A['x'], data_A['y'], data_A['dx'], data_A['dy'], fmt='None', color='black')
plt.plot(data_A['x'], llsrA.intercept + llsrA.slope*data_A['x'], 'r', label='Line of Best Fit', color='blue')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Dataset A')
plt.xlim(4,15)
plt.ylim(4,12)
plt.legend(frameon=False, loc=2)
plt.show()

    #**DATASET B**
# Using astropy.io to read Dataset B
data_B = ascii.read('Dataset B.txt')

print(data_B)

# Calculate the mean and standard deviation of y values in Dataset B
mean_y_B = np.mean(data_B['y'])
std_dev_y_B = np.std(data_B['y'])

# Print the calculated mean and standard deviation
print("Mean of y in Dataset B:", mean_y_B)
print("Standard Deviation of y in Dataset B:", std_dev_y_B)

# This creates a linear least-squares regression for x and y columns in data_B
llsrB = scipy.stats.linregress(data_B['x'], data_B['y'])

print(llsrB)

# Making a plot with error bars for Data_B

plt.figure(figsize=(7,4))
plt.plot(data_B['x'], data_B['y'], '.', label='Data B', color='black')
plt.errorbar(data_B['x'], data_B['y'], data_B['dx'], data_B['dy'], fmt='None', color='black')
plt.plot(data_B['x'], llsrB.intercept + llsrB.slope*data_B['x'], 'r', label='Line of Best Fit', color='red')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Dataset B')
plt.xlim(4,15)
plt.ylim(4,12)
plt.legend(frameon=False, loc=2)
plt.show()

    #**DATASET C**
# Using astropy.io to read Dataset C
data_C = ascii.read('Dataset C.txt')

print(data_C)

# Calculate the mean and standard deviation of y values in Dataset C
mean_y_C = np.mean(data_C['y'])
std_dev_y_C = np.std(data_C['y'])

# Print the calculated mean and standard deviation
print("Mean of y in Dataset C:", mean_y_C)
print("Standard Deviation of y in Dataset C:", std_dev_y_C)

# This creates a linear least-squares regression for x and y columns in data_C
llsrC = scipy.stats.linregress(data_C['x'], data_C['y'])

print(llsrC)

# Making a plot with error bars for Data_C

plt.figure(figsize=(7,4))
plt.plot(data_C['x'], data_C['y'], '.', label='Data C', color='black')
plt.errorbar(data_C['x'], data_C['y'], data_C['dx'], data_C['dy'], fmt='None', color='black')
plt.plot(data_C['x'], llsrC.intercept + llsrC.slope*data_C['x'], 'r', label='Line of Best Fit', color='green')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Dataset C')
plt.xlim(4,15)
plt.ylim(4,12)
plt.legend(frameon=False, loc=2)
plt.show()

    #**DATASET D**
# Using astropy.io to read Dataset D
data_D = ascii.read('Dataset D.txt')

print(data_D)

# Calculate the mean and standard deviation of y values in Dataset C
mean_y_D = np.mean(data_D['y'])
std_dev_y_D = np.std(data_D['y'])

# Print the calculated mean and standard deviation
print("Mean of y in Dataset D:", mean_y_D)
print("Standard Deviation of y in Dataset D:", std_dev_y_D)


# This creates a linear least-squares regression for x and y columns in data_D
llsrD = scipy.stats.linregress(data_D['x'], data_D['y'])

print(llsrD)

plt.figure(figsize=(7,4))
plt.plot(data_D['x'], data_D['y'], '.', label='Data D', color='black')
plt.errorbar(data_D['x'], data_D['y'], data_D['dx'], data_D['dy'], fmt='None', color='black')
plt.plot(data_D['x'], llsrD.intercept + llsrD.slope*data_D['x'], 'r', label='Line of Best Fit', color='purple')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Dataset D')
plt.xlim(7,20)
plt.ylim(4,13)
plt.legend(frameon=False, loc=2)
plt.show()



