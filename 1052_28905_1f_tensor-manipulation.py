import numpy as np
import psi4
import time

size = 20

if size > 30:
    raise Exception("Size must be smaller than 30.")
D = np.random.rand(size, size)
I = np.random.rand(size, size, size, size)

# Build the fock matrix using loops, while keeping track of time
tstart_loop = time.time()
Gloop = np.zeros((size, size))
for p in range(size):
    for q in range(size):
        for r in range(size):
            for s in range(size):
                Gloop[p, q] += 2 * I[p, q, r, s] * D[r, s]
                Gloop[p, q] -=     I[p, r, q, s] * D[r, s]

g_loop_time = time.time() - tstart_loop

# Build the fock matrix using einsum, while keeping track of time
tstart_einsum = time.time()
J = np.einsum('pqrs,rs', I, D)
K = np.einsum('prqs,rs', I, D)
G = 2 * J - K

einsum_time = time.time() - tstart_einsum

# Make sure the correct answer is obtained
print('The loop and einsum fock builds match:    %s\n' % np.allclose(G, Gloop))
# Print out relative times for explicit loop vs einsum Fock builds
print('Time for loop G build:   %14.4f seconds' % g_loop_time)
print('Time for einsum G build: %14.4f seconds' % einsum_time)
print('G builds with einsum are {:3.4f} times faster than Python loops!'.format(g_loop_time / einsum_time))

size = 200
A = np.random.rand(size, size)
B = np.random.rand(size, size)
C = np.random.rand(size, size)

# First compute the pair product
tmp_dot = np.dot(A, B)
tmp_einsum = np.einsum('ij,jk->ik', A, B)
print("Pair product allclose: %s" % np.allclose(tmp_dot, tmp_einsum))

D_dot = np.dot(A, B).dot(C)
D_einsum = np.einsum('ij,jk,kl->il', A, B, C)
print("Chain multiplication allclose: %s" % np.allclose(D_dot, D_einsum))

print("\nnp.dot time:")
get_ipython().magic('timeit np.dot(A, B).dot(C)')

print("\nnp.einsum time")
get_ipython().magic("timeit np.einsum('ij,jk,kl->il', A, B, C)")

print("np.einsum factorized time:")
get_ipython().magic("timeit np.einsum('ik,kl->il', np.einsum('ij,jk->ik', A, B), C)")

# Grab orbitals
size = 15
if size > 15:
    raise Exception("Size must be smaller than 15.")
    
C = np.random.rand(size, size)
I = np.random.rand(size, size, size, size)

# Numpy einsum N^8 transformation.
print("\nStarting Numpy's N^8 transformation...")
n8_tstart = time.time()
MO_n8 = np.einsum('pI,qJ,pqrs,rK,sL->IJKL', C, C, I, C, C)
n8_time = time.time() - n8_tstart
print("...transformation complete in %.3f seconds." % (n8_time))

# Numpy einsum N^5 transformation.
print("\n\nStarting Numpy's N^5 transformation with einsum...")
n5_tstart = time.time()
MO_n5 = np.einsum('pA,pqrs->Aqrs', C, I)
MO_n5 = np.einsum('qB,Aqrs->ABrs', C, MO_n5)
MO_n5 = np.einsum('rC,ABrs->ABCs', C, MO_n5)
MO_n5 = np.einsum('sD,ABCs->ABCD', C, MO_n5)
n5_time = time.time() - n5_tstart
print("...transformation complete in %.3f seconds." % n5_time)
print("\nN^5 %4.2f faster than N^8 algorithm!" % (n8_time / n5_time))
print("Allclose: %s" % np.allclose(MO_n8, MO_n5))

# Numpy GEMM N^5 transformation.
# Try to figure this one out!
print("\n\nStarting Numpy's N^5 transformation with dot...")
dgemm_tstart = time.time()
MO = np.dot(C.T, I.reshape(size, -1))
MO = np.dot(MO.reshape(-1, size), C)
MO = MO.reshape(size, size, size, size).transpose(1, 0, 3, 2)

MO = np.dot(C.T, MO.reshape(size, -1))
MO = np.dot(MO.reshape(-1, size), C)
MO = MO.reshape(size, size, size, size).transpose(1, 0, 3, 2)
dgemm_time = time.time() - dgemm_tstart
print("...transformation complete in %.3f seconds." % dgemm_time)
print("\nAllclose: %s" % np.allclose(MO_n8, MO))
print("N^5 %4.2f faster than N^8 algorithm!" % (n8_time / dgemm_time))



