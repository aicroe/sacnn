import numpy as np

from mlscratch.measurer import Measurer

def compute_labels_accuracy(confusion_mat):
    height, width = confusion_mat.shape
    assert height == width
    num_labels = height
    labels_accuracy = np.zeros(num_labels)

    for label in range(num_labels):
        matchs_count = np.sum(confusion_mat[:, label])
        asserts = confusion_mat[label, label]
        labels_accuracy[label] = asserts / (1 if matchs_count == 0 else matchs_count)

    return labels_accuracy

class ConfusionMatrixMeasurer(Measurer):

    def measure(self, result, expected):
        assert result.shape == expected.shape
        num_samples, num_labels = result.shape
        confusion_mat = np.zeros((num_labels, num_labels))

        for index in range(num_samples):
            confusion_mat[np.argmax(result[index]), np.argmax(expected[index])] += 1

        return confusion_mat
