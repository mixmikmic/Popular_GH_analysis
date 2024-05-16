"""Tutorial: Describing the implementation of density-fitted MP2 from an RHF reference"""

__author__    = "Dominic A. Sirianni"
__credit__    = ["Dominic A. Sirianni", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-24"

# ==> Import statements & Global Options <==
import psi4
import numpy as np

psi4.set_memory(int(2e9))
numpy_memory = 2
psi4.core.set_output_file('output.dat', False)

# ==> Options Definitions & SCF E, Wfn <==
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'df',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

# Number of Occupied orbitals & MOs
ndocc = scf_wfn.nalpha()
nmo = scf_wfn.nmo()
nvirt = nmo - ndocc

# Get orbital energies, cast into NumPy array, and separate occupied & virtual
eps = np.asarray(scf_wfn.epsilon_a())
e_ij = eps[:ndocc]
e_ab = eps[ndocc:]

# Get MO coefficients from SCF wavefunction
C = np.asarray(scf_wfn.Ca())
Cocc = C[:, :ndocc]
Cvirt = C[:, ndocc:]

# ==> Density Fitted ERIs <==
# Build auxiliar basis set
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "aug-cc-pVDZ")

# Build instance of Mints object
orb = scf_wfn.basisset()
mints = psi4.core.MintsHelper(orb)

# Build a zero basis
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

# Raw 3-index
Ppq = np.squeeze(mints.ao_eri(zero_bas, aux, orb, orb))

# Build and invert the Coulomb metric
metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
metric.power(-0.5, 1.e-14)
metric = np.squeeze(metric)

Qpq = np.einsum("QP,Ppq->Qpq", metric, Ppq)

# ==> Transform Qpq -> Qmo @ O(N^4) <==
Qmo = np.einsum('pi,Qpq->Qiq', C, Qpq)
Qmo = np.einsum('Qiq,qj->Qij', Qmo, C)

# Get Occupied-Virtual Block
Qmo = Qmo[:, :ndocc, ndocc:]

# ==> Build VV Epsilon Tensor <==
e_vv = e_ab.reshape(-1, 1) + e_ab

mp2_os_corr = 0.0
mp2_ss_corr = 0.0
for i in range(ndocc):
    # Get epsilon_i from e_ij
    e_i = e_ij[i]
    
    # Get 2d array Qa for i from Qov
    i_Qa = Qmo[:, i, :]
    
    for j in range(i, ndocc):
        # Get epsilon_j from e_ij
        e_j = e_ij[j]
        
        # Get 2d array Qb for j from Qov
        j_Qb = Qmo[:, j, :]
        
        # Compute 2d ERI array for fixed i,j from Qa & Qb
        ij_Iab = np.einsum('Qa,Qb->ab', i_Qa, j_Qb)

        # Compute energy denominator
        if i == j:
            e_denom = 1.0 / (e_i + e_j - e_vv)
        else:
            e_denom = 2.0 / (e_i + e_j - e_vv)

        # Compute SS & OS MP2 Correlation
        mp2_os_corr += np.einsum('ab,ab,ab->', ij_Iab, ij_Iab, e_denom)
        mp2_ss_corr += np.einsum('ab,ab,ab->', ij_Iab, ij_Iab - ij_Iab.T, e_denom)

# Compute MP2 correlation & total MP2 Energy
mp2_corr = mp2_os_corr + mp2_ss_corr
MP2_E = scf_e + mp2_corr

# ==> Compare to Psi4 <==
psi4.driver.p4util.compare_values(psi4.energy('mp2'), MP2_E, 6, 'MP2 Energy')



