from .arch import Arch
import tensorflow as tf


class BaseArch(Arch):
    def __init__(self,
                 input_tensor,                 
                 layer1_list_filters,
                 layer1_list_biases,
                 layer2_weights,
                 layer2_biases,
                 keep_prob=1):
        self.input_tensor = input_tensor
        self.layer1_list_filters = layer1_list_filters
        self.layer1_list_biases = layer1_list_biases
        self.layer2_weights = layer2_weights
        self.layer2_biases = layer2_biases
        self.keep_prob = keep_prob
        self.prediction = None
        self.logits = None

    def build(self):
        _, sentence_length, word_dimension, _ = self.input_tensor.get_shape().as_list()
        
        layer1_pool_activations = []
        reshape_size = 0
        for (layer1_filters, layer1_biases) in zip(self.layer1_list_filters, self.layer1_list_biases):
            filter_heigth, filter_width, _, num_filters = layer1_filters.get_shape().as_list()
            assert filter_width == word_dimension
            layer1_conv = tf.nn.conv2d(
                self.input_tensor,
                layer1_filters,
                strides=[1, 1, 1, 1],
                padding='VALID')
            layer1_activation = tf.nn.relu(layer1_conv + layer1_biases)
            layer1_pooling = tf.nn.max_pool(
                layer1_activation,
                [1, sentence_length - filter_heigth + 1, 1, 1], ## The whole conv out is max pooled
                strides=[1, 1, 1, 1],
                padding='VALID')
            ## Pool results have shape (1, 1, num_filters)
            reshape_size += num_filters
            layer1_pool_activations.append(layer1_pooling) 
        layer1_pooling = tf.concat(layer1_pool_activations, -1)
        layer1_reshape = tf.reshape(layer1_pooling, [-1, reshape_size])
        layer1_dropout = tf.nn.dropout(layer1_reshape, self.keep_prob)
        layer2_linear = tf.matmul(layer1_dropout, self.layer2_weights) + self.layer2_biases

        self.prediction = tf.nn.softmax(layer2_linear)
        self.logits = layer2_linear


    def initialize_variables(self, session):
        for layer1_filter, layer1_biases in zip(self.layer1_list_filters, self.layer1_list_biases):
            session.run(layer1_filter.initializer)
            session.run(layer1_biases.initializer)
        session.run(self.layer2_weights.initializer)
        session.run(self.layer2_biases.initializer)

    def _get_variables_as_list(self):
        return self.layer1_list_filters + self.layer1_list_biases + [self.layer2_weights, self.layer2_biases]
