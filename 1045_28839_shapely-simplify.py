# magic command to display matplotlib plots inline within the ipython notebook webpage
get_ipython().magic('matplotlib inline')

# import necessary modules
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from shapely.geometry import LineString
from time import time

# load the point data
df = pd.read_csv('data/summer-travel-gps-full.csv')
coordinates = df.as_matrix(columns=['lat', 'lon'])

# create a shapely line from the point data
line = LineString(coordinates)

# all points in the simplified object will be within the tolerance distance of the original geometry
tolerance = 0.015

# if preserve topology is set to False the much quicker Douglas-Peucker algorithm is used
# we don't need to preserve topology bc we just need a set of points, not the relationship between them
simplified_line = line.simplify(tolerance, preserve_topology=False)

print(line.length, 'line length')
print(simplified_line.length, 'simplified line length')
print(len(line.coords), 'coordinate pairs in full data set')
print(len(simplified_line.coords), 'coordinate pairs in simplified data set')
print(round(((1 - float(len(simplified_line.coords)) / float(len(line.coords))) * 100), 1), 'percent compressed')

# save the simplified set of coordinates as a new dataframe
lon = pd.Series(pd.Series(simplified_line.coords.xy)[1])
lat = pd.Series(pd.Series(simplified_line.coords.xy)[0])
si = pd.DataFrame({'lon':lon, 'lat':lat})
si.tail()

start_time = time()

# df_label column will contain the label of the matching row from the original full data set
si['df_label'] = None

# for each coordinate pair in the simplified set
for si_label, si_row in si.iterrows():    
    si_coords = (si_row['lat'], si_row['lon'])
    
    # for each coordinate pair in the original full data set
    for df_label, df_row in df.iterrows():
        
        # compare tuples of coordinates, if the points match, save this row's label as the matching one
        if si_coords == (df_row['lat'], df_row['lon']):
            si.loc[si_label, 'df_label'] = df_label
            break
            
print('process took %s seconds' % round(time() - start_time, 2))

si.tail()

# select the rows from the original full data set whose labels appear in the df_label column of the simplified data set
rs = df.loc[si['df_label'].dropna().values]

#rs.to_csv('data/summer-travel-gps-simplified.csv', index=False)
rs.tail()

# plot the final simplified set of coordinate points vs the original full set
plt.figure(figsize=(10, 6), dpi=100)
rs_scatter = plt.scatter(rs['lon'], rs['lat'], c='m', alpha=0.3, s=150)
df_scatter = plt.scatter(df['lon'], df['lat'], c='k', alpha=0.4, s=10)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Simplified set of coordinate points vs original full set')
plt.legend((rs_scatter, df_scatter), ('Simplified', 'Original'), loc='upper left')
plt.show()



