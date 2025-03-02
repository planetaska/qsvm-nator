# This file is for testing the Qiskit is working
# Should be removed in the future

import os
import numpy as np
from sklearn.datasets import make_blobs
# from qiskit.aqua.utils import split_dataset_to_data_and_labels # deprecated
from sklearn import svm
from utils import svm_utils
from matplotlib import pyplot as plt

# Import breast cancer data
from datasets.data_wdbc import breast_cancer

n = 2 # number of principal components kept
training_dataset_size = 20
testing_dataset_size = 10

# Generate pre-processed data from raw data
# Also generates a plot after DR with PCA
sample_Total, training_input, test_input, class_labels = breast_cancer(training_dataset_size, testing_dataset_size, n)

data_train, label_train = svm_utils.split_dataset_to_data_and_labels(training_input)
data_test, label_test = svm_utils.split_dataset_to_data_and_labels(test_input)

# print(data_train[0])
# print(data_train[1])
# print(test_input)
# test_set = np.concatenate((test_input['Benign'], test_input['Malignant']))
# print(test_set)


# Set up the global seed to ensure reproducibility
from qiskit_machine_learning.utils import algorithm_globals
algorithm_globals.random_seed = 12345

# Toggle for using simulator or real quantum computer
use_simulator = True

# Setup simulator or runtime service
from qiskit_aer.aerprovider import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

if use_simulator:
    backend = AerSimulator()
else:
    # Make sure you have saved your token by running save-runtime.py once.
    # If you did not previously save your credentials, use the following line instead:
    # service = QiskitRuntimeService(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>")
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)

# Setup quantum circuit
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit.primitives import StatevectorSampler as Sampler
# from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import z_feature_map

# Create the feature map
# NOTE: this is the map for encoding classical data into quantum data; NOT the feature map in ML
feature_map = z_feature_map(feature_dimension=2, reps=1, entanglement="linear")
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# Setup QSVM
# QSVC stands for Quantum Support Vector Classifier
from qiskit_machine_learning.algorithms import QSVC

qsvc = QSVC(quantum_kernel=kernel)

qsvc.fit(data_train[0], data_train[1])

qsvc_score = qsvc.score(data_test[0], data_test[1])

print(f"QSVC classification test score: {qsvc_score}")

test_set = np.concatenate((test_input['Benign'], test_input['Malignant']))
y_test = qsvc.predict(test_set)
print(y_test)
