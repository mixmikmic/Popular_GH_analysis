import psi4
import numpy as np

# Random number array
array = np.random.rand(5, 5)

psi4_matrix = psi4.core.Matrix.from_array(array)
new_array = np.array(psi4_matrix)

print("Allclose new_array, array:", np.allclose(new_array, array))

matrix = psi4.core.Matrix(3, 3)
print("Zero Psi4 Matrix:")
print(np.array(matrix))

matrix.np[:] = 1
print("\nMatrix updated to ones:")
print(np.array(matrix))

print(psi4.core.Matrix(3, 3).np)

mat = psi4.core.Matrix(3, 3)
print(mat.np)

# or
print(np.asarray(psi4.core.Matrix(3, 3)))

mat = psi4.core.Matrix(3, 3)
mat_view = np.asarray(mat)

mat_view[:] = np.random.random(mat.shape)
print(mat.np)

mat_view = np.zeros((3, 3))

# Mat is not updated as we replaced the mat_view with a new NumPy matrix.
print(mat.np)

arr = np.random.rand(5)
vec = psi4.core.Vector.from_array(arr)
print(vec.np)

