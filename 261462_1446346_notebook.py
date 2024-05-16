# Importing the fuzzy package
import fuzzy

# Exploring the output of fuzzy.nysiis

print(fuzzy.nysiis('yesterday'))
# Testing equivalence of similar sounding words
fuzzy.nysiis('tomorrow') == fuzzy.nysiis('tommorow')

# Importing the pandas module
import pandas as pd

# Reading in datasets/nytkids_yearly.csv, which is semicolon delimited.
author_df = pd.read_csv('datasets/nytkids_yearly.csv', delimiter=';')

# Looping through author_df['Author'] to extract the authors first names
first_name = []
for name in author_df['Author']:
    first_name.append(name.split()[0])

# Adding first_name as a column to author_df
author_df['first_name'] = first_name

# Checking out the first few rows of author_df
author_df.head()

# Importing numpy
import numpy as np

# Looping through author's first names to create the nysiis (fuzzy) equivalent
nysiis_name = []
for first_name in author_df['first_name']:
    tmp = fuzzy.nysiis(first_name)
    nysiis_name.append(tmp.split()[0])

# Adding first_name as a column to author_df
author_df['first_name'] = first_name
# Adding nysiis_name as a column to author_df
author_df['nysiis_name'] = nysiis_name

num_bananas_one = np.unique(author_df['first_name'])
lst1 = list(num_bananas_one)
num_bananas_one = np.asarray(lst1)

num_bananas_two = np.unique(author_df['nysiis_name'])
lst2 = list(num_bananas_two)
num_bananas_two = np.asarray(lst2)

# Printing out the difference between unique firstnames and unique nysiis_names:
print(str("Difference is" + str(num_bananas_one) + "," + str(num_bananas_two) + "."))

import pandas as pd
# Reading in datasets/babynames_nysiis.csv, which is semicolon delimited.
babies_df = pd.read_csv('datasets/babynames_nysiis.csv', delimiter = ';')

# Looping through babies_df to and filling up gender
gender = []
for idx, row in babies_df.iterrows():
    if row[1] > row[2]:
        gender.append('F')
    elif row[1] < row[2]:
        gender.append('M')
    elif row[1] == row[2]:
        gender.append('N')
    else:
        gender
# Adding a gender column to babies_df
babies_df['gender'] = pd.Series(gender)

# Printing out the first few rows of babies_df
print(babies_df.head(10))

# This function returns the location of an element in a_list.
# Where an item does not exist, it returns -1.
def locate_in_list(a_list, element):
   loc_of_name = a_list.index(element) if element in a_list else -1
   return(loc_of_name)

# Looping through author_df['nysiis_name'] and appending the gender of each
# author to author_gender.
author_gender = []
# ...YOUR CODE FOR TASK 5...
#print(author_df['nysiis_name'])
for idx in author_df['nysiis_name']:
   index = locate_in_list(list(babies_df['babynysiis']),idx)
   #print(index)
   if(index==-1): 
       author_gender.append('Unknown')
   else: 
       author_gender.append(list(babies_df['gender'])[index])

# Adding author_gender to the author_df
# ...YOUR CODE FOR TASK 5...
author_df['author_gender'] = author_gender 

# Counting the author's genders
# ...YOUR CODE FOR TASK 5...
author_df['author_gender'].value_counts()

# Creating a list of unique years, sorted in ascending order.
years = np.unique(author_df['Year'])
# Initializing lists
males_by_yr = []
females_by_yr = []
unknown_by_yr = []

# Looping through years to find the number of male, female and unknown authors per year
# ...YOUR CODE FOR TASK 6...
for yy in years:   
   males_by_yr.append(len( author_df[ (author_df['Year']==yy) & (author_df['author_gender']=='M')  ] ))
   females_by_yr.append(len( author_df[ (author_df['Year']==yy) & (author_df['author_gender']=='F')  ] ))
   unknown_by_yr.append(len( author_df[ (author_df['Year']==yy) & (author_df['author_gender']=='Unknown')  ] ))

# Printing out yearly values to examine changes over time
# ...YOUR CODE FOR TASK 6...
print(males_by_yr)
print(females_by_yr)
print(unknown_by_yr)

# Importing matplotlib
import matplotlib.pyplot as plt

# This makes plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Plotting the bar chart
plt.bar(unknown_by_yr, 'year')

# [OPTIONAL] - Setting a title, and axes labels
plt.title('Awesome!')
plt.xlabel('X')
plt.ylabel('Y')

# Creating a new list, where 0.25 is added to each year
years_shifted = [year + 0.25 for year in years]

# Plotting males_by_yr by year
plt.bar(males_by_yr, 'year', width = 0.25, color = 'lightblue')

# Plotting females_by_yr by years_shifted
plt.bar(females_by_yr, 'year_shifted', width = 0.25, color = 'pink')

# [OPTIONAL] - Adding relevant Axes labels and Chart Title
plt.title('Awesome!')
plt.xlabel('X')
plt.ylabel('Y')

