import numpy as np

from lib.singleton_decorator import singleton


@singleton()
class ConfusionMatrix(object):

    def compute(self, predicted_labels, real_labels):
        assert real_labels.shape == predicted_labels.shape
        num_samples, num_labels = real_labels.shape
        matrix = np.zeros((num_labels, num_labels))
        for index in range(num_samples):
            matrix[np.argmax(real_labels[index]), np.argmax(predicted_labels[index])] += 1
        return matrix

    def accuracy(self, matrix):
        height, width = matrix.shape
        assert height == width
        accuracy = np.zeros(width)
        for index in range(height):
            accuracy[index] = matrix[index, index] / (1 if np.sum(matrix[:, index]) == 0 else np.sum(matrix[:, index]))
        return accuracy
