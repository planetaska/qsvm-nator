import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# from utils_adhoc import ad_hoc_utils

# Set plot path
plot_dir = os.path.join("plots", "adhoc")
os.makedirs(plot_dir, exist_ok=True)

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add it to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
# import logger utility
from utils import ad_hoc_utils, Tee

# Redirect stdout to both console and file
sys.stdout = Tee(os.path.join(plot_dir, "ah1.log.txt"))

###
### Part: Prepare data

# Use the ad hoc dataset as described in the reference paper
# https://arxiv.org/pdf/1804.11326
from qiskit_machine_learning.datasets import ad_hoc_data

# Define the dataset dimension and get our train and test subsets
adhoc_dimension = 2
train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    training_size=20,
    test_size=5,
    n=adhoc_dimension,
    gap=0.3,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)

# This dataset is two-dimensional, the two features are represented by the x and y coordinates,
# and it has two class labels: A and B.
# Plot and see what the distribution looks like
data_plot_path = os.path.join(plot_dir, "ah1.01.adhoc_dataset.png")
ad_hoc_utils.plot_dataset(
    train_features, train_labels, test_features, test_labels, adhoc_total, data_plot_path
)

###
### Part: Classical SVM


###
### Part: Quantum SVM

# Set up the global seed to ensure reproducibility
from qiskit_machine_learning.utils import algorithm_globals
algorithm_globals.random_seed = 12345

# Toggle for using simulator or real quantum computer
use_simulator = True

# Setup simulator or runtime service
# from qiskit_aer.aerprovider import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

if use_simulator:
    from qiskit.primitives import StatevectorSampler as Sampler
    # backend = AerSimulator()
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
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")

fidelity = ComputeUncompute(sampler=sampler)

adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)

### Classification with SVC
# Quantum kernel as a callback function
from sklearn.svm import SVC

adhoc_svc = SVC(kernel=adhoc_kernel.evaluate)

adhoc_svc.fit(train_features, train_labels)

adhoc_score_callable_function = adhoc_svc.score(test_features, test_labels)

print(f"Callable kernel classification test score: {adhoc_score_callable_function}")

# Precomputed kernel matrix
adhoc_matrix_train = adhoc_kernel.evaluate(x_vec=train_features)
adhoc_matrix_test = adhoc_kernel.evaluate(x_vec=test_features, y_vec=train_features)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(
    np.asmatrix(adhoc_matrix_train), interpolation="nearest", origin="upper", cmap="Blues"
)
axs[0].set_title("Ad hoc training kernel matrix")

axs[1].imshow(np.asmatrix(adhoc_matrix_test), interpolation="nearest", origin="upper", cmap="Reds")
axs[1].set_title("Ad hoc testing kernel matrix")

# plt.show()
plt.savefig(os.path.join(plot_dir, "ah1.02.kernel_matrix.png"))

# Train the classifier by calling fit with the training matrix and training dataset.
# Once the model is trained, we evaluate it using the test matrix on the test dataset.
adhoc_svc = SVC(kernel="precomputed")

adhoc_svc.fit(adhoc_matrix_train, train_labels)

adhoc_score_precomputed_kernel = adhoc_svc.score(adhoc_matrix_test, test_labels)

print(f"Precomputed kernel classification test score: {adhoc_score_precomputed_kernel}")


### Classification with QSVC
# QSVC is an alternative training algorithm provided by qiskit-machine-learning for convenience.
# It is an extension of SVC that takes in a quantum kernel instead of the
# kernel.evaluate method shown before.

from qiskit_machine_learning.algorithms import QSVC

qsvc = QSVC(quantum_kernel=adhoc_kernel)

qsvc.fit(train_features, train_labels)

qsvc_score = qsvc.score(test_features, test_labels)

print(f"QSVC classification test score: {qsvc_score}")

# Evaluation of models used for classification
print(f"Classification Model                    | Accuracy Score")
print(f"---------------------------------------------------------")
print(f"SVC using kernel as a callable function | {adhoc_score_callable_function:10.2f}")
print(f"SVC using precomputed kernel matrix     | {adhoc_score_precomputed_kernel:10.2f}")
print(f"QSVC                                    | {qsvc_score:10.2f}")


# Restore original stdout when done
sys.stdout = sys.stdout.stdout
