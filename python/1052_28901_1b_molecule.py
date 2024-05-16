import psi4

he = psi4.geometry("""
He
""")

h2o = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
""")

doublet_h2o_cation = psi4.geometry("""
1 2
O
H 1 1.814
H 1 1.814 2 104.5

units bohr
symmetry c2v
""")

hydronium_benzene = psi4.geometry("""
0 1
C          0.710500000000    -0.794637665924    -1.230622098778
C          1.421000000000    -0.794637665924     0.000000000000
C          0.710500000000    -0.794637665924     1.230622098778
C         -0.710500000000    -0.794637665924     1.230622098778
H          1.254500000000    -0.794637665924    -2.172857738095
H         -1.254500000000    -0.794637665924     2.172857738095
C         -0.710500000000    -0.794637665924    -1.230622098778
C         -1.421000000000    -0.794637665924     0.000000000000
H          2.509000000000    -0.794637665924     0.000000000000
H          1.254500000000    -0.794637665924     2.172857738095
H         -1.254500000000    -0.794637665924    -2.172857738095
H         -2.509000000000    -0.794637665924     0.000000000000
-- 
1 1
X  1  CC  3  30   2  A2
O  13 R   1  90   2  90
H  14 OH  13 TDA  1  0
H  14 OH  15 TDA  13 A1
H  14 OH  15 TDA  13 -A1

CC    = 1.421
CH    = 1.088
A1    = 120.0
A2    = 180.0
OH    = 1.05
R     = 4.0
units angstrom
""")

h2cch2 = psi4.geometry("""
H
C 1 HC
H 2 HC 1 A1
C 2 CC 3 A1 1 D1
H 4 HC 2 A1 1 D1
H 4 HC 2 A1 1 D2

HC = 1.08
CC = 1.4
A1 = 120.0
D1 = 180.0
D2 = 0.0
""")

print("Ethene has %d atoms" % h2cch2.natom())


h2cch2.update_geometry()
print("Ethene has %d atoms" % h2cch2.natom())

# Define He Dimer
he_dimer = """
He
--
He 1 **R**
"""

distances = [2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0]
energies = []
for d in distances:
    # Build a new molecule at each separation
    mol = psi4.geometry(he_dimer.replace('**R**', str(d)))
    
    # Compute the Counterpoise-Corrected interaction energy
    en = psi4.energy('MP2/aug-cc-pVDZ', molecule=mol, bsse_type='cp')

    # Place in a reasonable unit, Wavenumbers in this case
    en *= 219474.6
    
    # Append the value to our list
    energies.append(en)

print("Finished computing the potential!")

import numpy as np

# Fit data in least-squares way to a -12, -6 polynomial
powers = [-12, -6]
x = np.power(np.array(distances).reshape(-1, 1), powers)
coeffs = np.linalg.lstsq(x, energies)[0]

# Build list of points
fpoints = np.linspace(2, 7, 50).reshape(-1, 1)
fdata = np.power(fpoints, powers)

fit_energies = np.dot(fdata, coeffs)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.xlim((2, 7))  # X limits
plt.ylim((-7, 2))  # Y limits
plt.scatter(distances, energies)  # Scatter plot of the distances/energies
plt.plot(fpoints, fit_energies)  # Fit data
plt.plot([0,10], [0,0], 'k-')  # Make a line at 0



