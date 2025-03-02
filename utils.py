import operator
import numpy as np
import matplotlib.pyplot as plt

class svm_utils:

    def make_meshgrid(x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
        return xx, yy


    def plot_contours(ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def split_dataset_to_data_and_labels(dataset, class_names=None):
        """
        Split dataset to data and labels numpy array

        If `class_names` is given, use the desired label to class name mapping,
        or create the mapping based on the keys in the dataset.

        Args:
            dataset (dict): {'A': numpy.ndarray, 'B': numpy.ndarray, ...}
            class_names (dict): class name of dataset, {class_name: label}

        Returns:
            Union(tuple(list, dict), list):
                List contains two arrays of numpy.ndarray type
                where the array at index 0 is data, an NxD array, and at
                index 1 it is labels, an Nx1 array, containing values in range
                0 to K-1, where K is the number of classes. The dict is a map
                {str: int}, mapping class name to label. The tuple of list, dict is returned
                when `class_names` is not None, otherwise just the list is returned.

        Raises:
            KeyError: data set invalid
        """
        data = []
        labels = []
        if class_names is None:
            sorted_classes_name = sorted(list(dataset.keys()))
            class_to_label = {k: idx for idx, k in enumerate(sorted_classes_name)}
        else:
            class_to_label = class_names
        sorted_label = sorted(class_to_label.items(), key=operator.itemgetter(1))
        for class_name, _ in sorted_label:
            values = dataset[class_name]
            for value in values:
                data.append(value)
                try:
                    labels.append(class_to_label[class_name])
                except Exception as ex:  # pylint: disable=broad-except
                    raise KeyError('The dataset has different class names to '
                                   'the training data. error message: {}'.format(ex)) from ex
        data = np.asarray(data)
        labels = np.asarray(labels)
        if class_names is None:
            return [data, labels], class_to_label
        else:
            return [data, labels]