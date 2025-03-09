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
sys.stdout = Tee(os.path.join(plot_dir, "ah2.log.txt"))

# Clustering using qiskit-machine-learning and the spectral clustering algorithm from scikit-learn.

###
### Part: Prepare data

# use the ad hoc dataset, but now generated with a higher gap of 0.6 (in adhoc_qsvc: 0.3)
# between the two classes.
# Note that clustering falls under the category of unsupervised machine learning,
# so a test dataset is not required.

from qiskit_machine_learning.datasets import ad_hoc_data

adhoc_dimension = 2
train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    training_size=25,
    test_size=0,
    n=adhoc_dimension,
    gap=0.6,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)

# Plot and see what the distribution looks like
# Plot the clustering dataset
plt.figure(figsize=(5, 5))
plt.ylim(0, 2 * np.pi)
plt.xlim(0, 2 * np.pi)
plt.imshow(
    np.asmatrix(adhoc_total).T,
    interpolation="nearest",
    origin="lower",
    cmap="RdBu",
    extent=[0, 2 * np.pi, 0, 2 * np.pi],
)

# A label plot
ad_hoc_utils.plot_features(plt, train_features, train_labels, 0, "s", "w", "b", "A")

# B label plot
ad_hoc_utils.plot_features(plt, train_features, train_labels, 1, "o", "w", "r", "B")

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
plt.title("Ad hoc dataset for clustering")

# plt.show()
plt.savefig(os.path.join(plot_dir, "ah2.01.adhoc_dataset.png"), bbox_inches='tight')

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
else:
    # Make sure you have saved your token by running save-runtime.py once.
    # If you did not previously save your credentials, use the following line instead:
    # service = QiskitRuntimeService(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>")
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)

    from qiskit_ibm_runtime import SamplerV2 as SamplerV2  # for IBM Runtime
    sampler = SamplerV2(backend)

# Setup quantum circuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")

fidelity = ComputeUncompute(sampler=sampler)

adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)

# With the current FidelityQuantumKernel class in qiskit-machine-learning,
# we can only use the latter option, so we precompute the kernel matrix by calling evaluate
# and visualize it as follows:

adhoc_matrix = adhoc_kernel.evaluate(x_vec=train_features)

plt.figure(figsize=(5, 5))
plt.imshow(np.asmatrix(adhoc_matrix), interpolation="nearest", origin="upper", cmap="Greens")
plt.title("Ad hoc clustering kernel matrix")
# plt.show()
plt.savefig(os.path.join(plot_dir, "ah2.02.kernel_matrix.png"))

# Next, we define a spectral clustering model and fit it using the precomputed kernel.
# Further, we score the labels using normalized mutual information,
# since we know the class labels a priori (before hand).

from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

adhoc_spectral = SpectralClustering(2, affinity="precomputed")

cluster_labels = adhoc_spectral.fit_predict(adhoc_matrix)

cluster_score = normalized_mutual_info_score(cluster_labels, train_labels)

print(f"Clustering score: {cluster_score}")

# Restore original stdout when done
sys.stdout = sys.stdout.stdout
