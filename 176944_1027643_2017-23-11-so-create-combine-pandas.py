import pandas as pd
import numpy as np

# working with series
#create a series
s = pd.Series(np.random.randn(5))
#create a dataframe column
df = pd.DataFrame(s, columns=['Column_1'])
df 

#sorting 
df.sort_values(by='Column_1')

#boolean indexing
#It returns all rows in column_name,
#that are less than 10
df[df['Column_1'] <= 1]

# creating simple series
obj2 = pd.Series(np.random.randn(5), index=['d', 'b', 'a', 'c', 'e'])
obj2

obj2.index

# returns the value in e
obj2['e']

# returns all values that are greater than -2
obj2[obj2 > -2]

# we can do multiplication on dataframe
obj2 * 2

# we can do boolean expression
'b' in obj2

# returns false, because 'g' is not defined in our data
'g' in obj2

#Let's see we have this data
sdata = {'Cat': 24, 'Dog': 11, 'Fox': 18, 'Horse': 1000}
obj3 = pd.Series(sdata)
obj3

# defined list, and assign series to it
sindex = ['Lion', 'Dog', 'Cat', 'Horse']
obj4 = pd.Series(sdata, index=sindex)
obj4

# checking if our data contains null
obj4.isnull()

#we can add two dataframe together
obj3 + obj4

# we can create series calling Series function on pandas
programming = pd.Series([89,78,90,100,98])
programming

# assign index to names
programming.index = ['C++', 'C', 'R', 'Python', 'Java']
programming

# let's create simple data
data = {'Programming': ['C++', 'C', 'R', 'Python', 'Java'],
        'Year': [1998, 1972, 1993, 1980, 1991],
        'Popular': [90, 79, 75, 99, 97]}
frame = pd.DataFrame(data)
frame

# set our index 
pd.DataFrame(data, columns=['Popular', 'Programming', 'Year'])

data2 = pd.DataFrame(data, columns=['Year', 'Programming', 'Popular', 'Users'],
                    index=[1,2,3,4,5])
data2

data2['Programming']

data2.Popular

data2.Users = np.random.random(5)*104
data2 = np.round(data2)
data2

# we will do merging two dataset together 
merg1 = {'Edit': 24, 'View': 11, 'Inser': 18, 'Cell': 40}
merg1 = pd.Series(merg1)
merg1 = pd.DataFrame(merg1, columns=['Merge1'])

merg2 = {'Kernel':50, 'Navigate':27, 'Widgets':29,'Help':43}
merg2 = pd.Series(merg2)
merg2 = pd.DataFrame(merg2, columns=['Merge2'])

merg1

merg2

#join matching rows from bdf to adf
#pd.merge(merg1, merg2, left_index=True, right_index=True)
join = merg1.join(merg2)
join

#replace all NA/null data with value
join = join.fillna(12)
join

#compute and append one or more new columns
join = join.assign(Area=lambda df: join.Merge1*join.Merge2)
join

#add single column
join['Volume'] = join.Merge1*join.Merge2*join.Area
join

join.head(2)

join.tail(2)

#randomly select fraction of rows
join.sample(frac=0.5)

#order rows by values of a column (low to high)
join.sort_values('Volume')

#order row by values of a column (high to low)
join.sort_values('Volume', ascending=False)

#return the columns of a dataframe - by renaming
join = join.rename(columns={'Merge1':'X','Merge2':'Y'})

join

#count number of rows with each unique value of variable
join['Y'].value_counts()

#number of rows in dataframe
len(join)

#descriptive statistics
join.describe()



