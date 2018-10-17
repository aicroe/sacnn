import tensorflow as tf
import numpy as np
import math

from lib.helpers.confusion_matrix import ConfusionMatrix
from .train_iterators.simple_iterator import SimpleIterator
from .helpers.accuracy import Accuracy


class SACNN(object):

    @staticmethod
    def generate_random_minibatches(inputs, outputs, minibatch_size):
        assert inputs.shape[0] == outputs.shape[0]
        batch_size = inputs.shape[0]
        permutation = np.random.permutation(batch_size)
        inputs_shuffled = inputs[permutation]
        outputs_shuffeld = outputs[permutation]
        num_batches = math.ceil(batch_size / minibatch_size)
        return zip(np.array_split(inputs_shuffled, num_batches), np.array_split(outputs_shuffeld, num_batches))

    def __init__(
            self,
            name,
            arch,
            graph,
            accuracy=Accuracy.get_instance(),
            confusion_matrix=ConfusionMatrix.get_instance()):
        self.name = name
        self.arch = arch
        self.graph = graph
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.arch.build()
            _, self.output_size = self.arch.logits.get_shape().as_list()
            self.expected_output = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size])

    def train(self,
              train_dataset,
              train_labels,
              hparams,
              epoch_print_cost,
              epoch_callback,
              val_dataset,
              val_labels,
              iterator=SimpleIterator.get_instance()):
        epochs = hparams.epochs
        minibatch_size = hparams.minibatch_size
        keep_prob = hparams.keep_prob
        learning_rate = hparams.learning_rate

        np.random.seed(0)
        tf.set_random_seed(0)
        with self.graph.as_default():
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.arch.logits, labels=self.expected_output))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        num_minibatches = math.ceil(train_dataset.shape[0] / minibatch_size)

        def perform_epoch():
            minibatch_cost = 0.
            minibatch_accuarcy = 0.
            minibatches = SACNN.generate_random_minibatches(train_dataset,
                                                            train_labels,
                                                            minibatch_size)
            for (minibatch_dataset, minibatch_labels) in minibatches:
                train_dict = {
                    self.arch.input_tensor: minibatch_dataset,
                    self.expected_output: minibatch_labels,
                    self.arch.keep_prob: keep_prob
                }
                target_tensors = [self.arch.prediction, optimizer, cost]
                train_predictions, _, current_cost = self.session.run(target_tensors, feed_dict=train_dict)
                minibatch_cost += current_cost / num_minibatches
                minibatch_accuarcy += self.accuracy(train_predictions, minibatch_labels) / num_minibatches
            return minibatch_cost, minibatch_accuarcy

        def evaluate_valset():
            val_dict = {
                self.arch.input_tensor: val_dataset,
                self.expected_output: val_labels,
                self.arch.keep_prob: 1
            }
            val_predictions = self.arch.prediction.eval(session=self.session, feed_dict=val_dict)
            val_cost = cost.eval(session=self.session, feed_dict=val_dict)
            return val_predictions, val_cost, val_labels

        self.arch.initialize_variables(self.session)
        return iterator.run(epochs, perform_epoch, epoch_print_cost, epoch_callback, evaluate_valset)

    def test(self, test_dataset, test_labels):
        test_dict = {
            self.arch.input_tensor: test_dataset,
            self.expected_output: test_labels,
            self.arch.keep_prob: 1
        }
        test_predictions = self.arch.prediction.eval(session=self.session, feed_dict=test_dict)
        with self.graph.as_default():
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.arch.logits, labels=test_labels))
        test_accuracy = self.accuracy(test_predictions, test_labels)
        test_cost = cost.eval(session=self.session, feed_dict=test_dict)
        confusion_matrix = self.confusion_matrix.compute(test_predictions, test_labels)
        return test_cost, test_accuracy, confusion_matrix

    def evaluate(self, input_dataset):
        eval_dict = {
            self.arch.input_tensor: input_dataset,
            self.arch.keep_prob: 1
        }
        predictions = self.arch.prediction.eval(session=self.session, feed_dict=eval_dict)
        return np.argmax(predictions, axis=1) + 1

    def save(self):
        return self.arch.save(self.session, self.name)

    def restore(self):
        return self.arch.restore(self.session, self.name)
