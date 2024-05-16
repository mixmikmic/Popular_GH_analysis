# ==> Define function to diagonalize F <==
def diag_F(F, norb):
    F_p = A.dot(F).dot(A)
    e, C_p = np.linalg.eigh(F_p)
    C = A.dot(C_p)
    C_occ = C[:, :norb]
    D = np.einsum('pi,qi->pq', C_occ, C_occ)
    return (C, D)

# ==> Build DIIS Extrapolation Function <==
def diis_xtrap(F_list, DIIS_RESID):
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
    F_DIIS = np.zeros_like(F_list[0])
    for x in range(coeff.shape[0] - 1):
        F_DIIS += coeff[x] * F_list[x]
    
    return F_DIIS

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
psi4.set_options({'guess': 'core',
                  'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'reference': 'uhf'})

# ==> Set default program options <==
# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-6
D_conv = 1.0e-3

# ==> Compute static 1e- and 2e- quantities with Psi4 <==
# Class instantiation
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap matrix
S = np.asarray(mints.ao_overlap())

# Number of basis Functions, alpha & beta orbitals, and # doubly occupied orbitals
nbf = wfn.nso()
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()
ndocc = min(nalpha, nbeta)

print('Number of basis functions: %d' % (nbf))
print('Number of singly occupied orbitals: %d' % (abs(nalpha - nbeta)))
print('Number of doubly occupied orbitals: %d' % (ndocc))

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

# Construct AO orthogonalization matrix A
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

# ==> Build alpha & beta CORE guess <==
Ca, Da = diag_F(H, nalpha)
Cb, Db = diag_F(H, nbeta)

# Get nuclear repulsion energy
E_nuc = mol.nuclear_repulsion_energy()

# ==> Pre-Iteration Setup <==
# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0

# Trial & Residual Vector Lists -- one each for alpha & beta
F_list_a = []
F_list_b = []
R_list_a = []
R_list_b = []

# ==> UHF-SCF Iterations <==
print('==> Starting SCF Iterations <==\n')

# Begin Iterations
for scf_iter in range(MAXITER):
    # Build Fa & Fb matrices
    Ja = np.einsum('pqrs,rs->pq', I, Da)
    Jb = np.einsum('pqrs,rs->pq', I, Db)
    Ka = np.einsum('prqs,rs->pq', I, Da)
    Kb = np.einsum('prqs,rs->pq', I, Db)
    Fa = H + (Ja + Jb) - Ka
    Fb = H + (Ja + Jb) - Kb
    
    # Compute DIIS residual for Fa & Fb
    diis_r_a = A.dot(Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)).dot(A)
    diis_r_b = A.dot(Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)).dot(A)
    
    # Append trial & residual vectors to lists
    F_list_a.append(Fa)
    F_list_b.append(Fb)
    R_list_a.append(diis_r_a)
    R_list_b.append(diis_r_b)
    
    # Compute UHF Energy
    SCF_E = np.einsum('pq,pq->', (Da + Db), H)
    SCF_E += np.einsum('pq,pq->', Da, Fa)
    SCF_E += np.einsum('pq,pq->', Db, Fb)
    SCF_E *= 0.5
    SCF_E += E_nuc
    
    dE = SCF_E - E_old
    dRMS = 0.5 * (np.mean(diis_r_a**2)**0.5 + np.mean(diis_r_b**2)**0.5)
    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E' % (scf_iter, SCF_E, dE, dRMS))
    
    # Convergence Check
    if (abs(dE) < E_conv) and (dRMS < D_conv):
        break
    E_old = SCF_E
    
    # DIIS Extrapolation
    if scf_iter >= 2:
        Fa = diis_xtrap(F_list_a, R_list_a)
        Fb = diis_xtrap(F_list_b, R_list_b)
    
    # Compute new orbital guess
    Ca, Da = diag_F(Fa, nalpha)
    Cb, Db = diag_F(Fb, nbeta)
    
    # MAXITER exceeded?
    if (scf_iter == MAXITER):
        psi4.core.clean()
        raise Exception("Maximum number of SCF iterations exceeded.")

# Post iterations
print('\nSCF converged.')
print('Final UHF Energy: %.8f [Eh]' % SCF_E)

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.driver.p4util.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')



