# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 01:54:16 2023

Charlie Ashe, 21365365
CompSci A.1.
"""

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# Opening file, each line to string, closing file
with open('planets.csv') as f:
    planets = f.readlines()

# Stripping \n from the end of each line & excluding comments
planets = [string.strip('\n') for string in planets if not string.startswith('#')]
# Splitting into lists at each comma
planets = [string.split(',') for string in planets]

# Listing years planets were discovered
disc_year = [int(entry[2]) for entry in planets[1:]]

# Counting the number of planets discovered per year
planet_counts = {}
for year in disc_year:
    planet_counts[year] = planet_counts.get(year, 0) + 1

# Sorting the data by year
years = sorted(planet_counts.keys())

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(years, [planet_counts[year] for year in years], color='indigo', marker='o')
plt.title('Number of Planets Discovered Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Planets Discovered')
plt.xlim(1988,2025)
plt.ylim(0,)
plt.grid(True)
plt.show()

# Listing planet masses and star masses (as floats)
planet_masses = [float(entry[5]) if entry[5] != '' else None for entry in planets[1:]]
star_masses = [float(entry[6]) if entry[6] != '' else None for entry in planets[1:]]

# Removing entries with missing data
data = list(zip(planet_masses, star_masses))
data = [(planet, star) for planet, star in data if planet is not None and star is not None]

# Unpacking zip 
planet_masses, star_masses = zip(*data)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(planet_masses, star_masses, color='darkred', marker='.')
plt.xscale('log')
plt.yscale('log')
plt.title('Planet Mass vs Star Mass')
plt.xlabel('Planet Mass [Jupiter Mass]')
plt.ylabel('Star Mass [Solar Mass]')
plt.grid(True)
plt.show()

# Listing facilities
facilities = [entry[3] for entry in planets[1:]]

# Counting number of planets discovered at each facility
facility_counts = {}
for facility in facilities:
    facility_counts[facility] = facility_counts.get(facility, 0) + 1

# Top ten facilities by the number of planets discovered
top_ten_facilities = sorted(facility_counts.keys(), key=facility_counts.get, reverse=True)[:10]

# Finding the year of first discovery for each of the top ten facilities
facility_first_year = {}
for entry in planets[1:]:
    facility = entry[3]
    year = int(entry[2])
    if facility in top_ten_facilities:
        if facility not in facility_first_year:
            facility_first_year[facility] = year
        else:
            facility_first_year[facility] = min(facility_first_year[facility], year)

# Creating list of tuples for the top ten facilities
facility_info = [(facility, facility_counts[facility], facility_first_year[facility]) for facility in top_ten_facilities]

# Sorting list by the year of first discovery (in ascending order)
sorted_facilities = sorted(facility_info, key=lambda x: x[2])

# Printing table header
print(f"{'Facility':<30}{'Total Planets Discovered':<25}{'Year of First Discovery':<25}")
print('-' * 80)

# Printing list of top ten facilities organized by the year of first discovery
for facility, total_planets, first_year in sorted_facilities:
    print(f"{facility:<30}{total_planets:<25}{first_year:<25}")

# Finding facility with the most planets discovered overall
most_planets_facility = max(facility_counts, key=facility_counts.get)
print(f"\nFacility with the Most Planets Discovered Overall: {most_planets_facility}, Total Planets: {facility_counts[most_planets_facility]}")
