import tensorflow as tf
from mlscratch.arch import Arch

from .saveable import Saveable, save_tf_instance, restore_tf_instance


class Kim1FcArch(Arch, Saveable):

    def __init__(
            self,
            name,
            sentence_length,
            word_dimension,
            filter_height,
            num_filters,
            hidden_units,
            num_labels,
            learning_rate,
            keep_prob=1):
        graph = tf.Graph()
        self._name = name
        self._session = tf.Session(graph=graph)
        self._train_keep_prob = keep_prob

        with graph.as_default():
            # Placeholders
            self._input = tf.placeholder(
                tf.float32,
                [None, sentence_length, word_dimension, 1],
            )
            self._keep_prob = tf.placeholder(tf.float32)
            self._expected_output = tf.placeholder(
                dtype=tf.float32,
                shape=[None, num_labels],
            )

            # Variables
            layer1_filters = tf.Variable(tf.truncated_normal(
                [filter_height, word_dimension, 1, num_filters],
                stddev=0.1,
            ))
            layer1_bias = tf.Variable(tf.zeros([num_filters]))
            layer2_weights = tf.Variable(tf.truncated_normal(
                [num_filters, hidden_units],
                stddev=0.1),
            )
            layer2_bias = tf.Variable(tf.zeros([hidden_units]))
            layer3_weights = tf.Variable(tf.truncated_normal(
                [hidden_units, num_labels],
                stddev=0.1),
            )
            layer3_bias = tf.Variable(tf.zeros([num_labels]))

            # Assemble
            layer1_conv = tf.nn.conv2d(
                self._input,
                layer1_filters,
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
            layer1_activation = tf.nn.relu(layer1_conv + layer1_bias)
            layer1_pooling = tf.nn.max_pool(
                layer1_activation,
                [1, sentence_length - filter_height + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
            layer1_reshape = tf.reshape(
                layer1_pooling,
                [-1, num_filters],
            )
            layer1_dropout = tf.nn.dropout(layer1_reshape, self._keep_prob)

            layer2_linear = tf.matmul(
                layer1_dropout,
                layer2_weights,
            ) + layer2_bias
            layer2_activation = tf.nn.relu(layer2_linear)
            layer2_dropout = tf.nn.dropout(layer2_activation, self._keep_prob)
            layer3_linear = tf.matmul(
                layer2_dropout,
                layer3_weights,
            ) + layer3_bias

            # Tensors
            self._prediction = tf.nn.softmax(layer3_linear)
            self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=layer3_linear,
                labels=self._expected_output,
            ))
            self._optimizer = tf.train.GradientDescentOptimizer(
                learning_rate
            ).minimize(self._cost)

        self._variables = [
            layer1_filters,
            layer1_bias,
            layer2_weights,
            layer2_bias,
            layer3_weights,
            layer3_bias,
        ]

    def get_name(self):
        return self._name

    def train_initialize(self):
        for variable in self._variables:
            self._session.run(variable.initializer)

    def train_finalize(self):
        pass

    def update_params(self, dataset, labels):
        feed_dict = {
            self._input: dataset,
            self._expected_output: labels,
            self._keep_prob: self._train_keep_prob,
        }
        _, cost, predictions = self._session.run(
            [self._optimizer, self._cost, self._prediction],
            feed_dict=feed_dict,
        )
        return cost, predictions

    def evaluate(self, dataset):
        feed_dict = {
            self._input: dataset,
            self._keep_prob: 1,
        }
        return self._prediction.eval(
            session=self._session,
            feed_dict=feed_dict,
        )

    def check_cost(self, dataset, labels):
        feed_dict = {
            self._input: dataset,
            self._expected_output: labels,
            self._keep_prob: 1,
        }
        return self._session.run(
            [self._cost, self._prediction],
            feed_dict=feed_dict,
        )

    def save(self):
        arch_name = type(self).__name__.lower()
        save_tf_instance(self._name, arch_name, self._variables, self._session)

    def restore(self):
        arch_name = type(self).__name__.lower()
        restore_tf_instance(self._name, arch_name, self._variables, self._session)
