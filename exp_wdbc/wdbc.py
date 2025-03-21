import os
import sys
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import svm
# from utils_wdbc import svm_utils

from matplotlib import pyplot as plt

# Import breast cancer data
# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add it to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
# import the dataset
from datasets.data_wdbc import breast_cancer
# import logger utility
from utils import svm_utils, Tee

# Set plot path
plot_dir = os.path.join("plots", "wdbc")
os.makedirs(plot_dir, exist_ok=True)

# Redirect stdout to both console and file
sys.stdout = Tee(os.path.join(plot_dir, "wdbc.log.txt"))

n = 2 # number of principal components kept
training_dataset_size = 20
testing_dataset_size = 10

# Generate pre-processed data from raw data
# Also generates a plot after DR with PCA
sample_Total, training_input, test_input, class_labels = breast_cancer(
    training_dataset_size, testing_dataset_size, n, plot_dir)

data_train, label_train = svm_utils.split_dataset_to_data_and_labels(training_input)
data_test, label_test = svm_utils.split_dataset_to_data_and_labels(test_input)

# We use the function of scikit learn to generate linearly separable blobs
centers = [(2.5,0),(0,2.5)]
x, y = make_blobs(n_samples=100, centers=centers, n_features=2,random_state=0,cluster_std=0.5)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(data_train[0][:,0],data_train[0][:,1],c=data_train[1])
ax[0].set_title('Breast Cancer dataset')

ax[1].scatter(x[:,0],x[:,1],c=y)
ax[1].set_title('Blobs linearly separable')

# Have a look at the dataset
plt.scatter(data_train[0][:, 0], data_train[0][:, 1], c=data_train[1])
plt.title('Breast Cancer dataset')
plt.savefig(os.path.join(plot_dir, "02.dataset_scatter.png"))

###
### Part: Classical SVM

# Define a linear SVM and train it on the dataset
model= svm.LinearSVC()
model.fit(data_train[0], data_train[1])

accuracy_train = model.score(data_train[0], data_train[1])
accuracy_test = model.score(data_test[0], data_test[1])

X0, X1 = data_train[0][:, 0], data_train[0][:, 1]
xx, yy = svm_utils.make_meshgrid(X0, X1)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].contourf(xx, yy, Z, cmap=plt.get_cmap('coolwarm'))
ax[0].scatter(data_train[0][:,0], data_train[0][:,1], c=data_train[1])
ax[0].set_title('Accuracy on the training set: '+str(accuracy_train))

ax[1].contourf(xx, yy, Z, cmap=plt.get_cmap('coolwarm'))
ax[1].scatter(data_test[0][:,0], data_test[0][:,1], c=data_test[1])
ax[1].set_title('Accuracy on the test set: '+str(accuracy_test))

plt.savefig(os.path.join(plot_dir, "03.linear_svm_results.png"))

# Implement a SVM with gaussian kernel
clf = svm.SVC(gamma = 'scale')
clf.fit(data_train[0], data_train[1])

accuracy_train = clf.score(data_train[0], data_train[1])
accuracy_test = clf.score(data_test[0], data_test[1])

X0, X1 = data_train[0][:, 0], data_train[0][:, 1]
xx, yy = svm_utils.make_meshgrid(X0, X1)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].contourf(xx, yy, Z, cmap=plt.get_cmap('coolwarm'))
ax[0].scatter(data_train[0][:,0], data_train[0][:,1], c=data_train[1])
ax[0].set_title('Accuracy on the training set: '+str(accuracy_train))

ax[1].contourf(xx, yy, Z, cmap=plt.get_cmap('coolwarm'))
ax[1].scatter(data_test[0][:,0], data_test[0][:,1], c=data_test[1])
ax[1].set_title('Accuracy on the test set: '+str(accuracy_test))

plt.savefig(os.path.join(plot_dir, "04.gaussian_svm_results.png"))

###
### Part: Quantum SVM

# Set up the global seed to ensure reproducibility
from qiskit_machine_learning.utils import algorithm_globals
algorithm_globals.random_seed = 12345

# Toggle for using simulator or real quantum computer
use_simulator = True

# Setup simulator or runtime service
from qiskit_aer.aerprovider import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

if use_simulator:
    from qiskit.primitives import StatevectorSampler as Sampler
    backend = AerSimulator()
    sampler = Sampler()
    print("Using quantum simulator")
else:
    # Make sure you have saved your token by running save-runtime.py once.
    # If you did not previously save your credentials, use the following line instead:
    # service = QiskitRuntimeService(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>")
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)

    from qiskit_ibm_runtime import SamplerV2 as SamplerV2  # for IBM Runtime
    sampler = SamplerV2(backend)
    print("Using real quantum computer")

# Setup quantum circuit
# from qiskit import QuantumCircuit
# from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import z_feature_map

# Create the feature map
# NOTE: this is the map for encoding classical data into quantum data; NOT the feature map in ML
feature_map = z_feature_map(feature_dimension=2, reps=1, entanglement="linear")
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# Setup QSVM
# QSVC stands for Quantum Support Vector Classifier
from qiskit_machine_learning.algorithms import QSVC

qsvc = QSVC(quantum_kernel=kernel)

qsvc.fit(data_train[0], data_train[1])

qsvc_score = qsvc.score(data_test[0], data_test[1])

print(f"QSVC classification test score: {qsvc_score}")


# Report and display comparison data
plt.scatter(training_input['Benign'][:, 0], training_input['Benign'][:, 1])
plt.scatter(training_input['Malignant'][:, 0], training_input['Malignant'][:, 1])
plt.savefig(os.path.join(plot_dir, "05.training_input_scatter.png"))
# plt.show()

length_data = len(training_input['Benign']) + len(training_input['Malignant'])
print("size training set: {}".format(length_data))
print("Matrix dimension: {}".format(data_train[0].shape))

print("testing success ratio: ", qsvc_score)

test_set = np.concatenate((test_input['Benign'], test_input['Malignant']))
y_test = qsvc.predict(test_set)

plt.scatter(test_set[:, 0], test_set[:, 1], c=y_test)
plt.savefig(os.path.join(plot_dir, "06.test_set_scatter.png"))
# plt.show()

plt.scatter(test_input['Benign'][:, 0], test_input['Benign'][:, 1])
plt.scatter(test_input['Malignant'][:, 0], test_input['Malignant'][:, 1])
plt.savefig(os.path.join(plot_dir, "07.test_input_scatter.png"))
# plt.show()

# Restore original stdout when done
sys.stdout = sys.stdout.stdout
