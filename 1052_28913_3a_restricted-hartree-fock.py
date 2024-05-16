# ==> Import Psi4 & NumPy <==
import psi4
import numpy as np

# ==> Set Basic Psi4 Options <==
# Memory specification
psi4.set_memory(int(5e8))
numpy_memory = 2

# Set output file
psi4.core.set_output_file('output.dat', False)

# Define Physicist's water -- don't forget C1 symmetry!
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# Set computation options
psi4.set_options({'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8})

# ==> Set default program options <==
# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-6

# ==> Compute static 1e- and 2e- quantities with Psi4 <==
# Class instantiation
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap matrix
S = np.asarray(mints.ao_overlap())

# Number of basis Functions & doubly occupied orbitals
nbf = S.shape[0]
ndocc = wfn.nalpha()

print('Number of occupied orbitals: %3d' % (ndocc))
print('Number of basis functions: %3d' % (nbf))

# Memory check for ERI tensor
I_size = (nbf**4) * 8.e-9
print('\nSize of the ERI tensor will be {:4.2f} GB.'.format(I_size))
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory                      limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Build ERI Tensor
I = np.asarray(mints.ao_eri())

# Build core Hamiltonian
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
H = T + V

# ==> Inspecting S for AO orthonormality <==
hope = np.allclose(S, np.eye(S.shape[0]))
print('\nDo we have any hope that our AO basis is orthonormal? %s!' % (hope))

# ==> Construct AO orthogonalization matrix A <==
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

# Check orthonormality
S_p = A.dot(S).dot(A)
new_hope = np.allclose(S_p, np.eye(S.shape[0]))

if new_hope:
    print('There is a new hope for diagonalization!')
else:
    print("Whoops...something went wrong. Check that you've correctly built the transformation matrix.")

# ==> Compute C & D matrices with CORE guess <==
# Transformed Fock matrix
F_p = A.dot(H).dot(A)

# Diagonalize F_p for eigenvalues & eigenvectors with NumPy
e, C_p = np.linalg.eigh(F_p)

# Transform C_p back into AO basis
C = A.dot(C_p)

# Grab occupied orbitals
C_occ = C[:, :ndocc]

# Build density matrix from occupied orbitals
D = np.einsum('pi,qi->pq', C_occ, C_occ)

# ==> Nuclear Repulsion Energy <==
E_nuc = mol.nuclear_repulsion_energy()

# ==> SCF Iterations <==
# Pre-iteration energy declarations
SCF_E = 0.0
E_old = 0.0

print('==> Starting SCF Iterations <==\n')

# Begin Iterations
for scf_iter in range(1, MAXITER + 1):
    # Build Fock matrix
    J = np.einsum('pqrs,rs->pq', I, D)
    K = np.einsum('prqs,rs->pq', I, D)
    F = H + 2*J - K
    
    # Compute RHF energy
    SCF_E = np.einsum('pq,pq->', (H + F), D) + E_nuc
    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E' % (scf_iter, SCF_E, SCF_E - E_old))
    
    # SCF Converged?
    if (abs(SCF_E - E_old) < E_conv):
        break
    E_old = SCF_E
    
    # Compute new orbital guess
    F_p =  A.dot(F).dot(A)
    e, C_p = np.linalg.eigh(F_p)
    C = A.dot(C_p)
    C_occ = C[:, :ndocc]
    D = np.einsum('pi,qi->pq', C_occ, C_occ)
    
    # MAXITER exceeded?
    if (scf_iter == MAXITER):
        psi4.core.clean()
        raise Exception("Maximum number of SCF iterations exceeded.")

# Post iterations
print('\nSCF converged.')
print('Final RHF Energy: %.8f [Eh]' % (SCF_E))

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.driver.p4util.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')



