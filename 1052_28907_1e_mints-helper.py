# ==> Setup <==
# Import statements
import psi4
import numpy as np

# Memory & Output file
psi4.set_memory(int(2e9))
numpy_memory = 2
psi4.core.set_output_file('output.dat', False)

# Molecule definition
h2o = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
""")

# Basis Set
psi4.set_options({'basis': 'cc-pvdz'})

# ==> Build MintsHelper Instance <==
# Build new wavefunction
wfn = psi4.core.Wavefunction.build(h2o, psi4.core.get_global_option('basis'))

# Initialize MintsHelper with wavefunction's basis set
mints = psi4.core.MintsHelper(wfn.basisset())

# ==> Integrals galore! <==
# AO Overlap
S = np.asarray(mints.ao_overlap())

# Number of basis functions
nbf = S.shape[0]

# Memory check
I_size = (nbf ** 4) * 8.e-9
print('Size of the ERI tensor will be %4.2f GB.' % (I_size))
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory                      limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Compute AO-basis ERIs
I = mints.ao_eri()

# Compute AO Core Hamiltonian
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
H = T + V

