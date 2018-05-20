from model import Parameters, HyperParameters, SACNNBase, DataManager
import numpy as np
import tensorflow as tf


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / predictions.shape[0])


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
    with tf.Session() as session:
        test_predictions = prediction.eval()
        print('accuarcy over test set: %f' % accuracy(test_predictions, test_labels))

if __name__ == '__main__':
    _main()