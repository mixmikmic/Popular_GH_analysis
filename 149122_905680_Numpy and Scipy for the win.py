import numpy as np

mat = mat = np.random.random_sample((125,125))

mat

mat[1,1]

mat[1,]

mat.tolist()

np.save('my_mat',mat)

new_mat = np.load('my_mat.npy')

new_mat == mat

test_eq = _

test_list = test_eq

all([ t for test in test_list for t in test])

from scipy.sparse import csc_matrix
import scipy.sparse as sp

row = np.array([0, 2, 2, 0, 1, 2])
col = np.array([0, 0, 1, 2, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6])
sparse_matrix = csc_matrix((data, (row, col)), shape=(3, 3))

row.shape == col.shape == data.shape

sp.issparse(sparse_matrix)

sp.issparse(mat)

new_mat_no_sparse = sparse_matrix.toarray()

sp.issparse(new_mat_no_sparse)

new_mat_no_sparse == sparse_matrix

all([ elem for row in _.tolist() for elem in row])

new_mat_no_sparse

