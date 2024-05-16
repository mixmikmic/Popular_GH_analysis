import numpy as np

a = np.array([[1,2,4],
              [2,5,3], 
              [7,8,9]])
A = np.mat(a)
A

A = np.mat('1,2,4;2,5,3;7,8,9')
A

a = np.array([[ 1, 2],
              [ 3, 4]])
b = np.array([[10,20], 
              [30,40]])

np.bmat('a,b;b,a')

x = np.array([[4], [2], [3]])
x

A * x

print(A * A.I)

print(A ** 3)



