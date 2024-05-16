# ==> Psi4 & NumPy options, Geometry Definition <==
import numpy as np
import psi4

# Set numpy defaults
np.set_printoptions(precision=5, linewidth=200, suppress=True)

# Set Psi4 memory & output options
psi4.set_memory(int(2e9))
psi4.core.set_output_file('output.dat', False)

# Geometry specification
mol = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
""")

# Psi4 options
psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'df',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10})

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))

# Build auxiliary basis set
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "aug-cc-pVDZ")

# ==> Build Density-Fitted Integrals <==
# Get orbital basis & build zero basis
orb = wfn.basisset()
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

# Build instance of MintsHelper
mints = psi4.core.MintsHelper(orb)

# Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
Ppq = mints.ao_eri(zero_bas, aux, orb, orb)

# Build & invert Coulomb metric, dimension (1, Naux, 1, Naux)
metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
metric.power(-0.5, 1.e-14)

# Remove excess dimensions of Ppq, & metric
Ppq = np.squeeze(Ppq)
metric = np.squeeze(metric)

# Build the Qso object
Qpq = np.einsum('QP,Ppq->Qpq', metric, Ppq)

# ==> Compute SCF Wavefunction, Density Matrix, & 1-electron H <==
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
D = scf_wfn.Da()
H = scf_wfn.H()

# Two-step build of J with Qpq and D
X_Q = np.einsum('Qpq,pq->Q', Qpq, D)
J = np.einsum('Qpq,Q->pq', Qpq, X_Q)

# Two-step build of K with Qpq and D
Z_Qqr = np.einsum('Qrs,sq->Qrq', Qpq, D)
K = np.einsum('Qpq,Qrq->pr', Qpq, Z_Qqr)

# Build F from SCF 1 e- Hamiltonian and our density-fitted J & K
F = H + 2 * J - K
# Get converged Fock matrix from converged SCF wavefunction
scf_F = scf_wfn.Fa()

if np.allclose(F, scf_F):
    print("Nicely done!! Your density-fitted Fock matrix matches Psi4!")
else:
    print("Whoops...something went wrong.  Try again!")





