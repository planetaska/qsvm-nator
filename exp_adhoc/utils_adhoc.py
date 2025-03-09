import numpy as np
import matplotlib.pyplot as plt

class ad_hoc_utils:
    @staticmethod
    def plot_features(ax, features, labels, class_label, marker, face, edge, label):
        # A train plot
        ax.scatter(
            # x coordinate of labels where class is class_label
            features[np.where(labels[:] == class_label), 0],
            # y coordinate of labels where class is class_label
            features[np.where(labels[:] == class_label), 1],
            marker=marker,
            facecolors=face,
            edgecolors=edge,
            label=label,
        )

    @staticmethod
    def plot_dataset(train_features, train_labels, test_features, test_labels, adhoc_total, plot_path):
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

        # A train plot
        ad_hoc_utils.plot_features(plt, train_features, train_labels, 0, "s", "w", "b", "A train")

        # B train plot
        ad_hoc_utils.plot_features(plt, train_features, train_labels, 1, "o", "w", "r", "B train")

        # A test plot
        ad_hoc_utils.plot_features(plt, test_features, test_labels, 0, "s", "b", "w", "A test")

        # B test plot
        ad_hoc_utils.plot_features(plt, test_features, test_labels, 1, "o", "r", "w", "B test")

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.title("Ad hoc dataset")

        # plt.show()
        plt.savefig(plot_path, bbox_inches='tight')
