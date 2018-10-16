import tensorflow as tf
from lib.singleton_decorator import singleton
from lib.archs.evolved_arch import EvolvedArch
from lib.sacnn import SACNN
from .sacnn_creator import SACNNCreator
from .sacnn_validator import SACNNValidator
from .sacnn_trainer import SACNNTrainer


@singleton()
class EvolvedCANNBuilder(SACNNCreator, SACNNValidator, SACNNTrainer):

    def __init__(self):
        super().__init__([
            'name',
            'sentence_length',
            'word_dimension',
            'hidden_units',
            'filters_size',
            'num_labels'])

    def create(self, hyperparams):
        self.validate_hparams(hyperparams)

        name = hyperparams['name']
        sentence_length = int(hyperparams['sentence_length'])
        word_dimension = int(hyperparams['word_dimension'])
        channels = self.channels
        hidden_units = int(hyperparams['hidden_units'])
        filters_size = hyperparams['filters_size']
        num_labels = int(hyperparams['num_labels'])

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
                [next_layer_height, hidden_units], stddev=0.1))
            layer2_biases = tf.Variable(tf.zeros([hidden_units]))
            layer3_weights = tf.Variable(tf.truncated_normal(
                [hidden_units, num_labels], stddev=0.1))
            layer3_biases = tf.Variable(tf.zeros([num_labels]))

            arch = EvolvedArch(input_placeholder,
                               layer1_list_filters,
                               layer1_list_biases,
                               layer2_weights,
                               layer2_biases,
                               layer3_weights,
                               layer3_biases,
                               keep_prob)
            return SACNN(name, arch, graph)
