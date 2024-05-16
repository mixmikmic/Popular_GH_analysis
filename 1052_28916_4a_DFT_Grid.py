import psi4
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib notebook')

# Set computatation options and molecule, any single atom will do.

mol = psi4.geometry("He")
psi4.set_options({'BASIS':                'CC-PVDZ',
                  'DFT_SPHERICAL_POINTS': 50,
                  'DFT_RADIAL_POINTS':    12})

basis = psi4.core.BasisSet.build(mol, "ORBITAL", "CC-PVDZ")
sup = psi4.driver.dft_funcs.build_superfunctional("PBE", True)[0]
Vpot = psi4.core.VBase.build(basis, sup, "RV")
Vpot.initialize()

x, y, z, w = Vpot.get_np_xyzw()
R = np.sqrt(x **2 + y ** 2 + z **2)

fig, ax = plt.subplots()
ax.scatter(x, y, c=w)
#ax.set_xscale('log')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mask = R > 8
p = ax.scatter(x[mask], y[mask], z[mask], c=w[mask], marker='o')
plt.colorbar(p)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

mol = psi4.geometry("""
 O
 H 1 1.1
 H 1 1.1 2 104
""")
mol.update_geometry()
psi4.set_options({'BASIS': '              CC-PVDZ',
                  'DFT_SPHERICAL_POINTS': 26,
                  'DFT_RADIAL_POINTS':    12})

basis = psi4.core.BasisSet.build(mol, "ORBITAL", "CC-PVDZ")
sup = psi4.driver.dft_funcs.build_superfunctional("PBE", True)[0]
Vpot = psi4.core.VBase.build(basis, sup, "RV")
Vpot.initialize()
x, y, z, w = Vpot.get_np_xyzw()
R = np.sqrt(x **2 + y ** 2 + z **2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mask = R > 0
p = ax.scatter(x[mask], y[mask], z[mask], c=w[mask], marker='o')
plt.colorbar(p)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')



