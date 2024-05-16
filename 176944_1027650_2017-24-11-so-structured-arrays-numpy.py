import numpy as np

a = np.array([1.0,2.0,3.0,4.0], np.float32)

# called the function view on our data
a.view(np.complex64)

# assign our to data to dtype
my_dtype = np.dtype([('mass', 'float32'), ('vol', 'float32')])

a.view(my_dtype)

my_data = np.array([(1,1), (1,2), (2,1), (1,3)], my_dtype)
print(my_data)

my_data[0]

my_data[0]['vol']

my_data['mass']

my_data.sort(order=('vol', 'mass'))
print(my_data)

person_dtype = np.dtype([('name', 'S10'), ('age', 'int'), ('weight', 'float')])

person_dtype.itemsize

people = np.empty((3,4), person_dtype)

people['age'] = [[33, 25, 47, 54],
                 [29, 61, 32, 27],
                 [19, 33, 18, 54]]

people['weight'] = [[135., 105., 255., 140.],
                    [154., 202., 137., 187.],
                    [188., 135., 88., 145.]]

print(people)

people[-1,-1]



