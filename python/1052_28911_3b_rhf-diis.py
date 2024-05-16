# ==> Basic Setup <==
# Import statements
import psi4
import numpy as np

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

# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-6
D_conv = 1.0e-3

# ==> Static 1e- & 2e- Properties <==
# Class instantiation
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap matrix
S = np.asarray(mints.ao_overlap())

# Number of basis Functions & doubly occupied orbitals
nbf = S.shape[0]
ndocc = wfn.nalpha()

print('Number of occupied orbitals: %d' % ndocc)
print('Number of basis functions: %d' % nbf)

# Memory check for ERI tensor
I_size = (nbf**4) * 8.e-9
print('\nSize of the ERI tensor will be %4.2f GB.' % I_size)
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

# ==> CORE Guess <==
# AO Orthogonalization Matrix
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

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

# Nuclear Repulsion Energy
E_nuc = mol.nuclear_repulsion_energy()

# ==> Pre-Iteration Setup <==
# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0

# Start from fresh orbitals
F_p =  A.dot(H).dot(A)
e, C_p = np.linalg.eigh(F_p)
C = A.dot(C_p)
C_occ = C[:, :ndocc]
D = np.einsum('pi,qi->pq', C_occ, C_occ)

# Trial & Residual Vector Lists
F_list = []
DIIS_RESID = []

# ==> SCF Iterations w/ DIIS <==
print('==> Starting SCF Iterations <==\n')

# Begin Iterations
for scf_iter in range(1, MAXITER + 1):
    # Build Fock matrix
    J = np.einsum('pqrs,rs->pq', I, D)
    K = np.einsum('prqs,rs->pq', I, D)
    F = H + 2*J - K
    
    # Build DIIS Residual
    diis_r = A.dot(F.dot(D).dot(S) - S.dot(D).dot(F)).dot(A)
    
    # Append trial & residual vectors to lists
    F_list.append(F)
    DIIS_RESID.append(diis_r)
    
    # Compute RHF energy
    SCF_E = np.einsum('pq,pq->', (H + F), D) + E_nuc
    dE = SCF_E - E_old
    dRMS = np.mean(diis_r**2)**0.5
    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E' % (scf_iter, SCF_E, dE, dRMS))
    
    # SCF Converged?
    if (abs(dE) < E_conv) and (dRMS < D_conv):
        break
    E_old = SCF_E
    
    if scf_iter >= 2:
        # Build B matrix
        B_dim = len(F_list) + 1
        B = np.empty((B_dim, B_dim))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        for i in range(len(F_list)):
            for j in range(len(F_list)):
                B[i, j] = np.einsum('ij,ij->', DIIS_RESID[i], DIIS_RESID[j])

        # Build RHS of Pulay equation 
        rhs = np.zeros((B_dim))
        rhs[-1] = -1
        
        # Solve Pulay equation for c_i's with NumPy
        coeff = np.linalg.solve(B, rhs)
        
        # Build DIIS Fock matrix
        F = np.zeros_like(F)
        for x in range(coeff.shape[0] - 1):
            F += coeff[x] * F_list[x]
    
    # Compute new orbital guess with DIIS Fock matrix
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
print('Final RHF Energy: %.8f [Eh]' % SCF_E)

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.driver.p4util.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')



