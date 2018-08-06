from .base_arch import BaseArch
from .evolved_arch import EvolvedArch
from .sacnn import SACNN
from .data_processor import DataProcessor
from .comment_classifier import CommentClassifier
from .data_saver import DataSaver
from .word_embedding import WordEmbedding
import tensorflow as tf


class ClassifierFactory(object):
    channels = 1

    @staticmethod
    def base_model(name,
                   sentence_length,
                   word_dimension,
                   channels,
                   filters_size,
                   num_labels):
        graph = tf.Graph()
        with graph.as_default():
            input_placeholder = tf.placeholder(tf.float32, [None, sentence_length, word_dimension, channels])
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
            layer2_weights = tf.Variable(tf.truncated_normal([next_layer_height, num_labels], stddev=0.1))
            layer2_biases = tf.Variable(tf.zeros([num_labels]))
            
            arch = BaseArch(input_placeholder,
                        layer1_list_filters,
                        layer1_list_biases,
                        layer2_weights,
                        layer2_biases,
                        keep_prob)
            return SACNN(name, arch, graph)

    @staticmethod
    def evolved_model(name,
                      sentence_length,
                      word_dimension,
                      channels,
                      filters_size,
                      hidden_units,
                      num_labels):
        graph = tf.Graph()
        with graph.as_default():
            input_placeholder = tf.placeholder(tf.float32, [None, sentence_length, word_dimension, channels])
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
            layer2_weights = tf.Variable(tf.truncated_normal([next_layer_height, hidden_units], stddev=0.1))
            layer2_biases = tf.Variable(tf.zeros([hidden_units]))
            layer3_weights = tf.Variable(tf.truncated_normal([hidden_units, num_labels], stddev=0.1))
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

    @staticmethod
    def evolved_classifier(name,
                           sentence_length,
                           filters_size,
                           hidden_units,
                           num_labels):
        embedding, word_dimension = WordEmbedding.get_instance()
        channels = ClassifierFactory.channels
        model = ClassifierFactory.evolved_model(name,
                                                sentence_length,
                                                word_dimension,
                                                channels,
                                                filters_size,
                                                hidden_units,
                                                num_labels)
        data_processor = DataProcessor(embedding.wv, sentence_length, channels)
        return CommentClassifier(model, data_processor)

    @staticmethod
    def base_classifier(name,
                        sentence_length,
                        filters_size,
                        num_labels):
        embedding, word_dimension = WordEmbedding.get_instance()
        channels = ClassifierFactory.channels
        model = ClassifierFactory.base_model(name,
                                             sentence_length,
                                             word_dimension,
                                             channels,
                                             filters_size,
                                             num_labels)
        data_processor = DataProcessor(embedding.wv, sentence_length, channels)
        return CommentClassifier(model, data_processor)
