import openfermion
import openfermionpyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator

import os
import timeit

import cudaq
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np


geometry = [('H', (0.0,0.0,0.0)), ('C', (0.0,0.0,1.203)), ('C', (0.0, 0.0, 2.406)), ('H', (0.0, 0.0, 3.609))]
basis = 'sto3g'
multiplicity = 1
charge = 0

molecule = openfermionpyscf.run_pyscf(openfermion.MolecularData(geometry, basis, multiplicity,charge))


molecular_hamiltonian = molecule.get_molecular_hamiltonian()

fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)

qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

spin_ham = cudaq.SpinOperator(qubit_hamiltonian)


electron_count = 10
qubit_count = 24

@cudaq.kernel
def kernel(qubit_num:int, electron_num:int, thetas: list[float]):
        qubits = cudaq.qvector(qubit_num)

        for i in range(electron_num):
                x(qubits[i])

        cudaq.kernels.uccsd(qubits, thetas, electron_num, qubit_num)

parameter_count = cudaq.kernels.uccsd_num_parameters(electron_count,qubit_count)


# Initial variational parameters.
np.random.seed(42)
x0 = np.random.normal(0, 1, parameter_count)



#cudaq.set_target("nvidia-mgpu")

start_time = timeit.default_timer()
for x in range(1):
    exp_val = cudaq.observe(kernel, spin_ham, qubit_count, electron_count, x0).expectation()
end_time = timeit.default_timer()

print(end_time-start_time)

