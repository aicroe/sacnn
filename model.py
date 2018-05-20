import numpy as np
import tensorflow as tf


class Parameters(object):
    def __init__(self,
                 layer1_filters,
                 layer1_biases,
                 layer2_weights,
                 layer2_biases,
                 layer3_weights,
                 layer3_biases):
        self.layer1_filters = layer1_filters
        self.layer1_biases = layer1_biases
        self.layer2_weights = layer2_weights
        self.layer2_biases = layer2_biases
        self.layer3_weights = layer3_weights
        self.layer3_biases = layer3_biases


class HyperParameters(object):
    def __init__(self,
                 filter_heigth,
                 filter_weight,
                 pool_stride,
                 learning_rate):
        self.filter_heigth = filter_heigth
        self.filter_weight = filter_weight
        self.pool_stride = pool_stride
        self.learning_rate = learning_rate


class SACNNBase(object):
    def __init__(self, parameters, hparameters, dataset, labels, keep_prob):
        self.parameters = parameters
        self.hparameters = hparameters
        self.dataset = dataset
        self.labels = labels
        self.keep_prob = keep_prob

        layer1_filters = self.parameters.layer1_filters
        layer1_biases = self.parameters.layer1_biases
        layer2_weights = self.parameters.layer2_weights
        layer2_biases = self.parameters.layer2_biases
        layer3_weights = self.parameters.layer3_weights
        layer3_biases = self.parameters.layer3_biases

        filter_heigth = self.hparameters.filter_heigth
        filter_weight = self.hparameters.filter_weight
        pool_stride = self.hparameters.pool_stride
        learning_rate = self.hparameters.learning_rate

        layer1_conv = tf.nn.conv2d(
            self.dataset,
            layer1_filters,
            strides=[1, 1, 1, 1],
            padding='SAME')
        layer1_activation = tf.nn.relu(layer1_conv + layer1_biases)
        layer1_pooling = tf.nn.max_pool(
            layer1_activation,
            [1, filter_heigth, filter_weight, 1],
            strides=[1, pool_stride, pool_stride, 1],
            padding='VALID')
        shape = layer1_pooling.get_shape().as_list()
        layer1_reshape = tf.reshape(layer1_pooling, [-1, shape[1] * shape[2] * shape[3]])
        layer1_dropout = tf.nn.dropout(layer1_reshape, self.keep_prob)
        layer2_linear = tf.matmul(layer1_dropout, layer2_weights) + layer2_biases
        layer2_activation = tf.nn.relu(layer2_linear)
        layer3_linear = tf.matmul(layer2_activation, layer3_weights) + layer3_biases
        self.prediction = tf.nn.softmax(layer3_linear)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=layer3_linear, labels=self.labels))
        if learning_rate != 0:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)


class DataManager(object):

    @staticmethod
    def save_data(train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels):
        np.save('data/train_dataset.npy', train_dataset)
        np.save('data/train_labels.npy', train_labels)
        np.save('data/val_dataset.npy', val_dataset)
        np.save('data/val_labels.npy', val_labels)
        np.save('data/test_dataset.npy', test_dataset)
        np.save('data/test_labels.npy', test_labels)

    @staticmethod
    def save_parameters(layer1_filters,
                        layer1_biases,
                        layer2_weights,
                        layer2_biases,
                        layer3_weights,
                        layer3_biases):
        np.save('data/layer1_filters.npy', layer1_filters)
        np.save('data/layer1_biases.npy', layer1_biases)
        np.save('data/layer2_weights.npy', layer2_weights)
        np.save('data/layer2_biases.npy', layer2_biases)
        np.save('data/layer3_weights.npy', layer3_weights)
        np.save('data/layer3_biases.npy', layer3_biases)

    @staticmethod
    def load_train_data():
        train_dataset = np.load('data/train_dataset.npy')
        train_labels = np.load('data/train_labels.npy')
        val_dataset = np.load('data/val_dataset.npy')
        val_labels = np.load('data/val_labels.npy')
        test_dataset = np.load('data/test_dataset.npy')
        test_labels = np.load('data/test_labels.npy')
        return train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels

    @staticmethod
    def load_test_data():
        test_dataset = np.load('data/test_dataset.npy')
        test_labels = np.load('data/test_labels.npy')
        return test_dataset, test_labels

    @staticmethod
    def load_parameters():
        layer1_filters = np.load('data/layer1_filters.npy')
        layer1_biases = np.load('data/layer1_biases.npy')
        layer2_weights = np.load('data/layer2_weights.npy')
        layer2_biases = np.load('data/layer2_biases.npy')
        layer3_weights = np.load('data/layer3_weights.npy')
        layer3_biases = np.load('data/layer3_biases.npy')
        return layer1_filters, layer1_biases, layer2_weights, layer2_biases, layer3_weights, layer3_biases
