import psi4
import numpy as np

psi4.set_output_file("output.dat", True)

# optional
psi4.core.IOManager.shared_object().set_default_path("/scratch")

psi4.set_memory(int(5e8))
numpy_memory = 2

psi4.geometry("""
O 0.0 0.0 0.0 
H 1.0 0.0 0.0
H 0.0 1.0 0.0
symmetry c1
""")

psi4.set_options({'basis': 'cc-pvdz'})

