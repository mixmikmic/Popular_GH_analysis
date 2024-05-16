import numpy as np
from scipy import linalg

def state(m=10):
    T = np.zeros((m, m), dtype=complex)
    T[0,0] = 0.6 + 1j
    idx = np.diag_indices(m-1)
    T[(idx[0]+1, idx[1])] = 1
    
    Q = np.eye(m)
    
    return T, Q

def direct_inverse(A, Q):
    n = A.shape[0]
    return np.linalg.inv(np.eye(n**2) - np.kron(A,A.conj())).dot(Q.reshape(Q.size, 1)).reshape(n,n)

def direct_solver(A, Q):
    return linalg.solve_discrete_lyapunov(A, Q)

# Example
from numpy.testing import assert_allclose
np.set_printoptions(precision=10)
T, Q = state(3)
sol1 = direct_inverse(T, Q)
sol2 = direct_solver(T, Q)

assert_allclose(sol1,sol2)

# Timings for m=1
T, Q = state(1)
get_ipython().magic('timeit direct_inverse(T, Q)')
get_ipython().magic('timeit direct_solver(T, Q)')

# Timings for m=5
T, Q = state(5)
get_ipython().magic('timeit direct_inverse(T, Q)')
get_ipython().magic('timeit direct_solver(T, Q)')

# Timings for m=10
T, Q = state(10)
get_ipython().magic('timeit direct_inverse(T, Q)')
get_ipython().magic('timeit direct_solver(T, Q)')

# Timings for m=50
T, Q = state(50)
get_ipython().magic('timeit direct_inverse(T, Q)')
get_ipython().magic('timeit direct_solver(T, Q)')

def bilinear1(A, Q):
    A = A.conj().T
    n = A.shape[0]
    eye = np.eye(n)
    B = np.linalg.inv(A - eye).dot(A + eye)
    res = linalg.solve_lyapunov(B.conj().T, -Q)
    return 0.5*(B - eye).conj().T.dot(res).dot(B - eye)

def bilinear2(A, Q):
    A = A.conj().T
    n = A.shape[0]
    eye = np.eye(n)
    AI_inv = np.linalg.inv(A + eye)
    B = (A - eye).dot(AI_inv)
    C = 2*np.linalg.inv(A.conj().T + eye).dot(Q).dot(AI_inv)
    return linalg.solve_lyapunov(B.conj().T, -C)

# Example:
T, Q = state(3)
sol3 = bilinear1(T, Q)
sol4 = bilinear2(T, Q)

assert_allclose(sol1,sol3)
assert_allclose(sol3,sol4)

# Timings for m=1
T, Q = state(1)
get_ipython().magic('timeit bilinear1(T, Q)')
get_ipython().magic('timeit bilinear2(T, Q)')

# Timings for m=5
T, Q = state(5)
get_ipython().magic('timeit bilinear1(T, Q)')
get_ipython().magic('timeit bilinear2(T, Q)')

# Timings for m=10
T, Q = state(10)
get_ipython().magic('timeit bilinear1(T, Q)')
get_ipython().magic('timeit bilinear2(T, Q)')

# Timings for m=50
T, Q = state(50)
get_ipython().magic('timeit bilinear1(T, Q)')
get_ipython().magic('timeit bilinear2(T, Q)')

# Timings for m=500
T, Q = state(500)
get_ipython().magic('timeit bilinear1(T, Q)')
get_ipython().magic('timeit bilinear2(T, Q)')

