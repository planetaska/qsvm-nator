# This file is for testing the Qiskit is working
# Should be removed in the future

# Toggle for using simulator or real quantum computer
use_simulator = True

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.aerprovider import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService

if use_simulator:
    backend = AerSimulator()
else:
    # Make sure you have saved your token by running save-runtime.py once.
    # If you did not previously save your credentials, use the following line instead:
    # service = QiskitRuntimeService(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>")
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)

# Create a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)
# Add Hadamard Gate on the first Qubit
qc.h(0)
# CNOT Gate on the first and second Qubits
qc.cx(0, 1)
# Measure the qubits
qc.measure([0, 1], [0, 1])
# Draw the quantum circuit
qc.draw("mpl")

# Simulating the circuit using the simulator to get the result
# circuit optimization
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_qc = pm.run(qc)

# run with sampler
sampler = Sampler(backend)
job = sampler.run([isa_qc])
result = job.result()

# show the result
counts = result[0].data.c.get_counts()
print(f" > Counts: {counts}")

from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt
plot_histogram(counts)
plt.show()
