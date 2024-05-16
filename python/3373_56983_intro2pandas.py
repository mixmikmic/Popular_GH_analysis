import pandas as pd
import numpy as np

# We label the random values
s1 = pd.Series(np.random.rand(6), index=['q', 'w', 'e', 'r', 't', 'y'])

print(s1)

d = {'q': 11, 'w': 21, 'e': 31, 'r': 41}

# We pick out the q, w, and r keys, but have an undefined y key.
s2 = pd.Series(d, index = ['q', 'w', 'r', 'y'])

print(s2)

# We create a Series from an integer constant with explicit labels
s3 = pd.Series(42, index = ['q', 'w', 'e', 'r', 't', 'y'])

print(s3)

print('\nThe "e" value is ', s3['e'])

# We can slice like NumPy arrays

print(s1[:-2])

# We can also perform vectorized operations
print('\nSum Series:')
print(s1 + s3)
print('\nSeries operations:')
print(s2 * 5 - 1.2)

# Read data from CSV file, and display subset

dfa = pd.read_csv('data.csv', delimiter='|', index_col='iata')

# We can grab the first five rows, and only extract the 'airport' column
print(dfa[['airport', 'city', 'state']].head(5))

# Read data from our JSON file
dfb = pd.read_json('data.json')

# Grab the last five rows
print(dfb[[0, 1, 2, 3, 5, 6]].tail(5))

# Lets look at the datatypes of each column
dfa.dtypes

# We can get a summary of numerical information in the dataframe

dfa.describe()

# Notice the JSON data did not automatically specify data types
dfb.dtypes

# This affects the output of the describe method, dfb has no numerical data types.

dfb.describe()

# We can slice out rows using the indicated index for dfa

print(dfa.loc[['00V', '11R', '12C']])

# We can slice out rows using the row index for dfb
print(dfb[100:105])

# We can also select rows based on boolean tests on columns
print(dfa[(dfa.lat > 48) & (dfa.long < -170)])

df = pd.DataFrame(np.random.randn(5, 6))

print(df)

# We can incorporate operate with basic scalar values

df + 2.5

# And perform more complex scalar operations

-1.0 * df + 3.5

# We can also apply vectorized functions

np.sin(df)

# We can tranpose the dataframe

df.T

