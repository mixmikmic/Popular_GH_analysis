# Load packages
from pygeocoder import Geocoder
import pandas as pd
import numpy as np

# Create a dictionary of raw data
data = {'Site 1': '31.336968, -109.560959',
        'Site 2': '31.347745, -108.229963',
        'Site 3': '32.277621, -107.734724',
        'Site 4': '31.655494, -106.420484',
        'Site 5': '30.295053, -104.014528'}

# Convert the dictionary into a pandas dataframe
df = pd.DataFrame.from_dict(data, orient='index')

# View the dataframe
df

# Create two lists for the loop results to be placed
lat = []
lon = []

# For each row in a varible,
for row in df[0]:
    # Try to,
    try:
        # Split the row by comma, convert to float, and append
        # everything before the comma to lat
        lat.append(float(row.split(',')[0]))
        # Split the row by comma, convert to float, and append
        # everything after the comma to lon
        lon.append(float(row.split(',')[1]))
    # But if you get an error
    except:
        # append a missing value to lat
        lat.append(np.NaN)
        # append a missing value to lon
        lon.append(np.NaN)

# Create two new columns from lat and lon
df['latitude'] = lat
df['longitude'] = lon

# View the dataframe
df

# Convert longitude and latitude to a location
results = Geocoder.reverse_geocode(df['latitude'][0], df['longitude'][0])

# Print the lat/long
results.coordinates

# Print the city
results.city

# Print the country
results.country

# Print the street address (if applicable)
results.street_address

# Print the admin1 level
results.administrative_area_level_1

# Verify that an address is valid (i.e. in Google's system)
Geocoder.geocode("4207 N Washington Ave, Douglas, AZ 85607").valid_address

# Print the lat/long
results.coordinates

# Find the lat/long of a certain address
result = Geocoder.geocode("7250 South Tucson Boulevard, Tucson, AZ 85756")

# Print the street number
result.street_number

# Print the street name
result.route

