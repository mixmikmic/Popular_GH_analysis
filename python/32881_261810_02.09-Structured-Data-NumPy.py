import numpy as np

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

x = np.zeros(4, dtype=int)

# Use a compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

# Get all names
data['name']

# Get first row of data
data[0]

# Get the name from the last row
data[-1]['name']

# Get names where age is under 30
data[data['age'] < 30]['name']

np.dtype({'names':('name', 'age', 'weight'),
          'formats':('U10', 'i4', 'f8')})

np.dtype({'names':('name', 'age', 'weight'),
          'formats':((np.str_, 10), int, np.float32)})

np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])

np.dtype('S10,i4,f8')

tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0])

data['age']

data_rec = data.view(np.recarray)
data_rec.age

get_ipython().magic("timeit data['age']")
get_ipython().magic("timeit data_rec['age']")
get_ipython().magic('timeit data_rec.age')

