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
sys.stdout = Tee(os.path.join(plot_dir, "ah3.log.txt"))

# This section focuses on a Principal Component Analysis task using a kernel PCA algorithm.
# We calculate a kernel matrix using a ZZFeatureMap and show that this approach translates
# the original features into a new space, where axes are chosen along principal components.
# In this space the classification task can be performed with a simpler model rather than an SVM.

###
### Part: Prepare data

# Use the ad hoc dataset with a gap of 0.6 between the two classes.
# This dataset resembles the dataset we had in the clustering section,
# the difference is that in this case test_size is not zero.

from qiskit_machine_learning.datasets import ad_hoc_data

adhoc_dimension = 2
train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    training_size=25,
    test_size=10,
    n=adhoc_dimension,
    gap=0.6,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)

# Plot and see what the distribution looks like

# Plot the training and test datasets below. Ultimate goal in this section is to construct
# new coordinates where the two classes can be linearly separated.
data_plot_path = os.path.join(plot_dir, "ah3.01.adhoc_dataset.png")
ad_hoc_utils.plot_dataset(
    train_features, train_labels, test_features, test_labels, adhoc_total, data_plot_path
)

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

# Define the Quantum Kernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement="linear")
fidelity = ComputeUncompute(sampler=sampler)
qpca_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# Evaluate kernel matrices for the training and test features
matrix_train = qpca_kernel.evaluate(x_vec=train_features)
matrix_test = qpca_kernel.evaluate(x_vec=test_features, y_vec=train_features)

### Comparison of Kernel PCA on gaussian and quantum kernel

# In this section we use the KernelPCA implementation from scikit-learn,
# with the kernel parameter set to “rbf” for a gaussian kernel and “precomputed” for a quantum kernel.
# The former is very popular in classical machine learning models,
# whereas the latter allows using a quantum kernel defined as qpca_kernel.
#
# One can observe that the gaussian kernel based Kernel PCA model fails to make the dataset
# linearly separable, while the quantum kernel succeeds.

# While usually PCA is used to reduce the number of features in a dataset,
# or in other words to reduce dimensionality of a dataset, we don’t do that here.
# Rather we keep the number of dimensions and employ the kernel PCA, mostly for visualization purposes,
# to show that classification on the transformed dataset becomes easily tractable by linear methods,
# like logistic regression. We use this method to separate two classes in the principal component space
# with a LogisticRegression model from scikit-learn.

# Train the model by calling the fit method on the training dataset and
# evaluate the model for accuracy with score.
from sklearn.decomposition import KernelPCA

kernel_pca_rbf = KernelPCA(n_components=2, kernel="rbf")
kernel_pca_rbf.fit(train_features)
train_features_rbf = kernel_pca_rbf.transform(train_features)
test_features_rbf = kernel_pca_rbf.transform(test_features)

kernel_pca_q = KernelPCA(n_components=2, kernel="precomputed")
train_features_q = kernel_pca_q.fit_transform(matrix_train)
test_features_q = kernel_pca_q.transform(matrix_test)

# Train and score a model
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
logistic_regression.fit(train_features_q, train_labels)

logistic_score = logistic_regression.score(test_features_q, test_labels)
print(f"Logistic regression score: {logistic_score}")

# Plot the results.
# First, we plot the transformed dataset we get with the quantum kernel.
# On the same plot we also add model results.
# Then, we plot the transformed dataset we get with the gaussian kernel.

fig, (q_ax, rbf_ax) = plt.subplots(1, 2, figsize=(10, 5))


ad_hoc_utils.plot_features(q_ax, train_features_q, train_labels, 0, "s", "w", "b", "A train")
ad_hoc_utils.plot_features(q_ax, train_features_q, train_labels, 1, "o", "w", "r", "B train")

ad_hoc_utils.plot_features(q_ax, test_features_q, test_labels, 0, "s", "b", "w", "A test")
ad_hoc_utils.plot_features(q_ax, test_features_q, test_labels, 1, "o", "r", "w", "A test")

q_ax.set_ylabel("Principal component #1")
q_ax.set_xlabel("Principal component #0")
q_ax.set_title("Projection of training and test data\n using KPCA with Quantum Kernel")

# Plotting the linear separation
h = 0.01  # step size in the mesh

# create a mesh to plot in
x_min, x_max = train_features_q[:, 0].min() - 1, train_features_q[:, 0].max() + 1
y_min, y_max = train_features_q[:, 1].min() - 1, train_features_q[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

predictions = logistic_regression.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
predictions = predictions.reshape(xx.shape)
q_ax.contourf(xx, yy, predictions, cmap=plt.cm.RdBu, alpha=0.2)

ad_hoc_utils.plot_features(rbf_ax, train_features_rbf, train_labels, 0, "s", "w", "b", "A train")
ad_hoc_utils.plot_features(rbf_ax, train_features_rbf, train_labels, 1, "o", "w", "r", "B train")
ad_hoc_utils.plot_features(rbf_ax, test_features_rbf, test_labels, 0, "s", "b", "w", "A test")
ad_hoc_utils.plot_features(rbf_ax, test_features_rbf, test_labels, 1, "o", "r", "w", "A test")

rbf_ax.set_ylabel("Principal component #1")
rbf_ax.set_xlabel("Principal component #0")
rbf_ax.set_title("Projection of training data\n using KernelPCA")
# plt.show()
plt.savefig(os.path.join(plot_dir, "ah3.02.comparison.png"))

# As we can see, the data points on the right figure are not separable,
# but they are on the left figure, hence in case of quantum kernel we can apply linear models
# on the transformed dataset and this is why SVM classifier works perfectly well on the ad hoc dataset
# as we saw in the classification section.

# Restore original stdout when done
sys.stdout = sys.stdout.stdout
