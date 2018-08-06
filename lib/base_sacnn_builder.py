from .sacnn_builder import SACNNBuilder
from .base_arch import BaseArch
from .sacnn import SACNN
import tensorflow as tf


class BaseSCANNBuilder(SACNNBuilder):
    def __init__(self):
        super().__init__([
            'name',
            'sentence_length',
            'word_dimension',
            'filters_size',
            'num_labels'])

    def create(self, hyperparams):
        self.validate_hparams(hyperparams)

        name = hyperparams['name']
        sentence_length = hyperparams['sentence_length']
        word_dimension = hyperparams['word_dimension']
        channels = self.channels
        filters_size = hyperparams['filters_size']
        num_labels = hyperparams['num_labels']
        graph = tf.Graph()

        with graph.as_default():
            input_placeholder = tf.placeholder(
                tf.float32, [None, sentence_length, word_dimension, channels])
            keep_prob = tf.placeholder(tf.float32)

            layer1_list_filters = []
            layer1_list_biases = []
            next_layer_height = 0
            for (filter_height, num_filters) in filters_size:
                layer1_filters = tf.Variable(tf.truncated_normal([
                    filter_height, word_dimension, channels, num_filters], stddev=0.1))
                layer1_biases = tf.Variable(tf.zeros([num_filters]))
                next_layer_height += num_filters
                layer1_list_filters.append(layer1_filters)
                layer1_list_biases.append(layer1_biases)
            layer2_weights = tf.Variable(tf.truncated_normal(
                [next_layer_height, num_labels], stddev=0.1))
            layer2_biases = tf.Variable(tf.zeros([num_labels]))

            arch = BaseArch(input_placeholder,
                            layer1_list_filters,
                            layer1_list_biases,
                            layer2_weights,
                            layer2_biases,
                            keep_prob)
            return SACNN(name, arch, graph)
