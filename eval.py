from model import Parameters, HyperParameters, SACNNBase, DataManager
import numpy as np
import tensorflow as tf


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / predictions.shape[0])


def confusion_matrix(predicted_labels, real_labels):
    assert real_labels.shape == predicted_labels.shape
    num_samples, num_labels = real_labels.shape
    matrix = np.zeros((num_labels, num_labels))
    for index in range(num_samples):
        matrix[np.argmax(real_labels[index]), np.argmax(predicted_labels[index])] += 1
    return matrix

def confusion_matrix_accuracy(matrix):
    height, width = matrix.shape
    assert height == width
    accuracy = np.zeros(width)
    for index in range(height):
        accuracy[index] = matrix[index, index] / np.sum(matrix[:, index])
    return accuracy


def _main():
    test_dataset, test_labels = DataManager.load_test_data()

    (layer1_filters,
    layer1_biases,
    layer2_weights,
    layer2_biases,
    layer3_weights,
    layer3_biases) = DataManager.load_parameters()

    parameters = Parameters(
        tf.constant(layer1_filters),
        tf.constant(layer1_biases),
        tf.constant(layer2_weights),
        tf.constant(layer2_biases),
        tf.constant(layer3_weights),
        tf.constant(layer3_biases))

    filter_size = 5
    pool_stride = 5

    hparameters = HyperParameters(
        filter_size,
        filter_size,
        pool_stride,
        learning_rate=0)
    
    model = SACNNBase(
        parameters,
        hparameters,
        tf.constant(test_dataset),
        tf.constant(test_labels),
        keep_prob=1)

    prediction = model.prediction
    cost = model.cost
    with tf.Session() as session:
        test_predictions = prediction.eval()
        print('accuarcy over test set: %f' % accuracy(test_predictions, test_labels))
        print('cost over test set: %f' % cost.eval())
        matrix = confusion_matrix(test_predictions, test_labels)
        print('Confusion matrix:\n', matrix)
        print('accuracy:', confusion_matrix_accuracy(matrix))

if __name__ == '__main__':
    _main()
