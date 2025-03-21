import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions

# Set up the global seed to ensure reproducibility
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 12345

# Toggle for using simulator or real quantum computer
use_simulator = True

# Setup simulator or runtime service
from qiskit_aer.aerprovider import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

if use_simulator:
    print("Using quantum simulator")
    from qiskit.primitives import StatevectorSampler as Sampler

    backend = AerSimulator()
    sampler = Sampler()
else:
    print("Using real quantum computer")
    # Make sure you have saved your token by running save-runtime.py once.
    # If you did not previously save your credentials, use the following line instead:
    # service = QiskitRuntimeService(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>")
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)

    from qiskit_ibm_runtime import SamplerV2 as SamplerV2  # for IBM Runtime

    sampler = SamplerV2(backend)

from qiskit.circuit.library import z_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Setup quantum circuit
from qiskit_machine_learning.state_fidelities import ComputeUncompute

# Setup QSVM
from qiskit_machine_learning.algorithms import QSVC

# Number of principal components kept
n = 2

if __name__ == "__main__":
    # Set plot path
    plot_dir = os.path.join("plots", "esc")
    os.makedirs(plot_dir, exist_ok=True)

    # Get the absolute path of the project root and add it to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from utils import svm_utils, Tee
    from datasets.data_esc import esc

    # Redirect stdout to both console and file
    sys.stdout = Tee(os.path.join(plot_dir, "esc.log.txt"))

    # Load data
    training_input, test_input, class_labels = esc(dataset="esc10", n=n)

    data_train, label_train = svm_utils.split_dataset_to_data_and_labels(training_input)
    data_test, label_test = svm_utils.split_dataset_to_data_and_labels(test_input)

    # Number of principal components kept
    print(f"Number of principal components kept: n={n} (dimensions)")

    # Define a linear SVM and train it on the dataset
    clf = svm.LinearSVC()
    print("Training linear SVM")
    clf.fit(data_train[0], data_train[1])
    lin_acc_train = clf.score(data_train[0], data_train[1])
    lin_acc_test = clf.score(data_test[0], data_test[1])
    print(f"Accuracy on the training set: {lin_acc_train:.2f}")
    print(f"Accuracy on the test set: {lin_acc_test:.2f}")

    fig = plot_decision_regions(X=data_test[0], y=data_test[1], clf=clf, legend=2)
    plt.title("Linear SVM")
    plt.savefig(os.path.join(plot_dir, "01.esc_linear_svm_results.png"))
    # plt.show()

    # Define a Gaussian SVM and train it on the dataset
    clf = svm.SVC(kernel="rbf", gamma="scale")
    print("Training Gaussian SVM")
    clf.fit(data_train[0], data_train[1])
    gauss_acc_train = clf.score(data_train[0], data_train[1])
    gauss_acc_test = clf.score(data_test[0], data_test[1])
    print(f"Accuracy on the training set: {gauss_acc_train:.2f}")
    print(f"Accuracy on the test set: {gauss_acc_test:.2f}")

    fig = plot_decision_regions(X=data_test[0], y=data_test[1], clf=clf, legend=2)
    plt.title("Gaussian SVM")
    plt.savefig(os.path.join(plot_dir, "02.esc_gaussian_svm_results.png"))
    # plt.show()

    # Create the feature map
    feature_map = z_feature_map(feature_dimension=n, reps=1, entanglement="linear")
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    print("Quantum feature map, fidelity, and kernel created")

    qsvc = QSVC(quantum_kernel=kernel)
    print("Fitting QSVC")
    qsvc.fit(data_train[0], data_train[1])
    qsvc_acc_train = qsvc.score(data_train[0], data_train[1])
    qsvc_acc_test = qsvc.score(data_test[0], data_test[1])
    print(f"Accuracy on the training set: {qsvc_acc_train:.2f}")
    print(f"Accuracy on the test set: {qsvc_acc_test:.2f}")

    fig = plot_decision_regions(X=data_test[0], y=data_test[1], clf=clf, legend=2)
    plt.title("QSVM")
    plt.savefig(os.path.join(plot_dir, "03.esc_quantum_svm_results.png"))
    # plt.show()

