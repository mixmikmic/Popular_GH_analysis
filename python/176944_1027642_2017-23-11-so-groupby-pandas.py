# import library
import pandas as pd

data = {'Students': ['S1', 'S2', 'S3', 'S3', 'S1',
         'S4', 'S4', 'S3', 'S2', 'S2', 'S4', 'S3'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Grade':[87,79,83,73,74,81,56,78,94,70,80,69]}
df = pd.DataFrame(data)
df

# let's groupby students
df.groupby('Students')

# to view groups 
df.groupby('Students').groups

# you can group by with multiple columns 
df.groupby(['Students','Year']).groups

# iterating through groups
grouped = df.groupby('Students')
for student, group_name in grouped:
    print(student)
    print(group_name)

# select group by value
grouped = df.groupby('Year')
print(grouped.get_group(2014))

# find the mean of grouped by data
import numpy as np
grouped = df.groupby('Year')
print(grouped['Grade'].agg(np.mean))

# find the average for all students
grouped = df.groupby('Students')
print(grouped['Grade'].agg(np.mean).round())

# count 
grouped = df.groupby('Year')
print(grouped['Grade'].value_counts())

#Filtration filters the data on a defined criteria and returns the subset of data. 
#The filter() function is used to filter the data.
# we are going to find the top 3 students
df.groupby('Students').filter(lambda x: len(x) >= 3)



