import math
import numpy as np

size = 100000
delta = 1.0E-2

aList = [(x + delta) for x in range(size)]
anArray = np.arange(size) + delta

print(aList[2:6])
print(anArray[2:6])

get_ipython().magic('timeit [math.sin(x) for x in aList]')
get_ipython().magic('timeit [math.cos(x) for x in aList]')
get_ipython().magic('timeit [math.log(x) for x in aList]')

get_ipython().magic('timeit np.sin(anArray)')
get_ipython().magic('timeit np.cos(anArray)')
get_ipython().magic('timeit np.log10(anArray)')

l = [math.sin(x) for x in aList]
a = np.sin(anArray)

print("Python List: ", l[2:10:3])
print("NumPY array:", a[2:10:3])

print("Difference = ", a[5:7] - np.array(l[5:7]))
      
# Now create a NumPy array from a Python list
get_ipython().magic('timeit (np.sin(aList))')

# Make and print out simple NumPy arrays

print(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

print("\n", np.empty(10))
print("\n", np.zeros(10))
print("\n", np.ones(10))
print("\n", np.ones_like(np.arange(10)))

# Demonstrate the np.arange method

print(np.arange(10))
print(np.arange(3, 10, 2))

# Demonstrate linear and logarthmic array creation.

print("Linear space bins [0, 10] = {}\n".format(np.linspace(0, 10, 4)))

print("Default linspace bins = {}\n".format(len(np.linspace(0,10))))


print("Log space bins [0, 1] = {}\n".format(np.logspace(0, 1, 4)))

print("Default linspace bins = {}\n".format(len(np.logspace(0,10))))

# Access our previously created array's data type 

a.dtype

# Try to assign a string to a floating point array element

a[0] = 'Hello!'

# Make a 10 x 10 array

data = np.arange(100)

mat = data.reshape(10, 10)

print(mat)

# Create special two-dimensional arrays

print("Matrix will be 4 x 4.\n", np.eye(4))
print("\nMatrix will be 4 x 4.\n", np.diag(np.arange(4), 0))
print("\nMatrix will be 5 x 5.\n", np.diag(np.arange(4), 1))
print("\nMatrix will be 5 x 5.\n", np.diag(np.arange(4), -1))

a = np.arange(9)
print("Original Array = ", a)

a[1] = 3
a[3:5] = 4
a[0:6:2] *= -1

print("\nNew Array = ", a)

b = np.arange(9).reshape((3,3))

print("3 x 3 array = \n",b)

print("\nSlice in first dimension (row 1): ",b[0])
print("\nSlice in first dimension (row 3): ",b[2])

print("\nSlice in second dimension (col 1): ",b[:,0])
print("\nSlice in second dimension (col 3): ", b[:,2])

print("\nSlice in first and second dimension: ", b[0:1, 1:2])


print("\nDirect Element access: ", b[0,1])

c = np.arange(27).reshape((3,3, 3))

print("3 x 3 x 3 array = \n",c)
print("\nSlice in first dimension (first x axis slice):\n",c[0])

print("\nSlice in first and second dimension: ", c[0, 1])

print("\nSlice in first dimension (third x axis slice):\n", c[2])

print("\nSlice in second dimension (first y axis slice):\n", c[:,0])
print("\nSlice in second dimension (third y axis slice):\n", c[:,2])

print("\nSlice in first and second dimension: ", c[0:1, 1:2])

print("\nSlice in first and second dimension:\n", c[0,1])
print("\nSlice in first and third dimension: ", c[0,:,1])
print("\nSlice in first, second, and third dimension: ", c[0:1,1:2,2:])

print("\nDirect element access: ", c[0,1, 2])

# Demonstration of an index array

a = np.arange(10)

print("\nStarting array:\n", a)
print("\nIndex Access: ", a[np.array([1, 3, 5, 7])])

c = np.arange(10).reshape((2, 5))

print("\nStarting array:\n", c)
print("\nIndex Array access: \n", c[np.array([0, 1]) , np.array([3, 4])])

# Demonstrate Boolean mask access

# Simple case

a = np.arange(10)
print("Original Array:", a)

print("\nMask Array: ", a > 4)

# Now change the values by using the mask

a[a > 4] = -1.0
print("\nNew Array: ", a)

# Now a more complicated example.

print("\n--------------------")
c = np.arange(25).reshape((5, 5))
print("\n Starting Array: \n", c)

# Build a mask that is True for all even elements with value greater than four
mask1 = (c > 4)
mask2 = (c % 2 == 0)

print("\nMask 1:\n", mask1)
print("\nMask 2:\n", mask2)

# We use the logical_and ufunc here, but it is described later
mask = np.logical_and(mask1, mask2)

print("\nMask :\n", mask)

print("\nMasked Array :\n", c[mask])
c[mask] /= -2.

print("\nNew Array :\n", c)

# Create arrays of random data from the uniform distribution

print("Uniform sampling [0, 1): ", np.random.rand(5))
print("Uniform sampling, integers [0, 1): ", np.random.randint(0, 10, 5))
print("Normal sampling (0, 1) : ", np.random.randn(5))

# Create and use a vector
a = np.arange(10)

print(a)
print("\n", (2.0 * a + 1)/3)
print("\n", a%2)
print("\n", a//2)

# Create a two-dimensional array

b = np.arange(9).reshape((3,3))

print("Matrix = \n", b)

print("\nMatrix + 10.5 =\n", (b + 10.5))

print("\nMatrix * 0.25 =\n", (b * 0.25))

print("\nMatrix % 2 =\n", (b % 2))

print("\nMatrix / 3.0 =\n", ((b - 4.0) / 3.))

# Create two arrays

a = np.arange(1, 10)
b = (10. - a).reshape((3, 3))
print("Array = \n",a)
print("\nMatrix = \n",b)

print("\nArray[0:3] + Matrix Row 1 = ",a[:3] + b[0,:,])

print("\nArray[0:3] + Matrix[:0] = ", a[:3] + b[:,0])

print("\nArray[3:6] + Matrix[0:] = ", a[3:6] + b[0, :])

# Now combine scalar operations

print("\n3.0 * Array[3:6] + (10.5 + Matrix[0:]) = ", 3.0 * a[3:6] + (10.5 + b[0, :]))

# Demonstrate data processing convenience functions

# Make an array = [1, 2, 3, 4, 5]
a = np.arange(1, 6)

print("Mean value = {}".format(np.mean(a)))
print("Median value = {}".format(np.median(a)))
print("Variance = {}".format(np.var(a)))
print("Std. Deviation = {}\n".format(np.std(a)))

print("Minimum value = {}".format(np.min(a)))
print("Maximum value = {}\n".format(np.max(a)))

print("Sum of all values = {}".format(np.sum(a)))
print("Running cumulative sum of all values = {}\n".format(np.cumsum(a)))

print("Product of all values = {}".format(np.prod(a)))
print("Running product of all values = {}\n".format(np.cumprod(a)))

# Now compute trace of 5 x 5 diagonal matrix (= 5)
print(np.trace(np.eye(5)))

b = np.arange(1, 10).reshape(3, 3)

print('original array:\n', b)

c = np.sin(b)

print('\nnp.sin : \n', c)

print('\nnp.log and np.abs : \n', np.log10(np.abs(c)))

print('\nnp.mod : \n', np.mod(b, 2))

print('\nnp.logical_and : \n', np.logical_and(np.mod(b, 2), True))


# Demonstrate Boolean tests with operators

d = np.arange(9).reshape(3, 3)

print("Greater Than or Equal Test: \n", d >= 5)

# Now combine to form Boolean Matrix

np.logical_and(d > 3, d % 2)

# Create and demonstrate a masked array

import numpy.ma as ma

x = [0.,1.,-9999.,3.,4.]
print("Original array = :", x)


mx = ma.masked_values (x, -9999.)
print("\nMasked array = :", mx)


print("\nMean value of masked elements:", mx.mean())
print("\nOperate on unmaksed elements: ", mx - mx.mean())
print("\n Impute missing values (using mean): ", mx.filled(mx.mean())) # Imputation

# Create two arrays with masks
x = ma.array([1., -1., 3., 4., 5., 6.], mask=[0,0,0,0,1,0])
y = ma.array([1., 2., 0., 4., 5., 6.], mask=[0,0,0,0,0,1])

# Now take square root, ignores div by zero and masked elements.
print(np.sqrt(x/y))

# Now try some random data

d = np.random.rand(1000)

# Now mask for values within some specified range (0.1 to 0.9)
print("Masked array mean value: ", ma.masked_outside(d, 0.1, 0.9).mean())

# First write data to a file using Unix commands. 
info = "1, 2, 3, 4, 5 \n 6, 7, 8, 9, 10"
with open("test.csv", 'w') as fout:
    print(info, file=fout)

# Now we can read it back into a NumPy array.
d = np.genfromtxt("test.csv", delimiter=",")

print("New Array = \n", d)

np.lookfor('masked array')

