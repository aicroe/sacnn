import tensorflow as tf
import numpy as np
import math


class SACNN(object):

    @staticmethod
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / predictions.shape[0])

    @staticmethod
    def compute_confusion(predicted_labels, real_labels):
        assert real_labels.shape == predicted_labels.shape
        num_samples, num_labels = real_labels.shape
        matrix = np.zeros((num_labels, num_labels))
        for index in range(num_samples):
            matrix[np.argmax(real_labels[index]), np.argmax(predicted_labels[index])] += 1
        return matrix

    @staticmethod
    def confusion_matrix_accuracy(matrix):
        height, width = matrix.shape
        assert height == width
        accuracy = np.zeros(width)
        for index in range(height):
            accuracy[index] = matrix[index, index] / (1 if np.sum(matrix[:, index]) == 0 else np.sum(matrix[:, index]))
        return accuracy

    @staticmethod
    def generate_random_minibatches(inputs, outputs, minibatch_size):
        assert inputs.shape[0] == outputs.shape[0]
        batch_size = inputs.shape[0]
        permutation = np.random.permutation(batch_size)
        inputs_shuffled = inputs[permutation]
        outputs_shuffeld = outputs[permutation]
        num_batches = math.ceil(batch_size / minibatch_size)
        return zip(np.array_split(inputs_shuffled, num_batches), np.array_split(outputs_shuffeld, num_batches))

    def __init__(self, name, arch, graph):
        self.name = name
        self.arch = arch
        self.graph = graph
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
              val_labels):
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

        costs = []
        val_costs = []
        num_minibatches = math.ceil(train_dataset.shape[0] / minibatch_size)

        self.arch.initialize_variables(self.session)
        for epoch in range(epochs):
            minibatch_cost = 0.
            minibatch_accuarcy = 0.
            minibatches = self.generate_random_minibatches(train_dataset,
                                                           train_labels,
                                                           minibatch_size)
            for (minibatch_dataset, minibatch_labels) in minibatches:
                train_dict = {
                    self.arch.input_tensor: minibatch_dataset,
                    self.expected_output: minibatch_labels,
                    self.arch.keep_prob: keep_prob
                }
                tensors = [self.arch.prediction, optimizer, cost]
                train_predictions, _, current_cost = self.session.run(tensors, feed_dict=train_dict)
                minibatch_cost += current_cost / num_minibatches
                minibatch_accuarcy += self.accuracy(train_predictions, minibatch_labels) / num_minibatches

            costs.append(minibatch_cost)

            if epoch_print_cost > 0 and epoch % epoch_print_cost == 0:
                val_dict = {
                    self.arch.input_tensor: val_dataset,
                    self.expected_output: val_labels,
                    self.arch.keep_prob: 1
                }
                val_predictions = self.arch.prediction.eval(session=self.session, feed_dict=val_dict)
                val_cost = cost.eval(session=self.session, feed_dict=val_dict)
                val_costs.append(val_cost)
                epoch_callback(int(epoch),
                               float(minibatch_accuarcy),
                               float(minibatch_cost),
                               float(self.accuracy(val_predictions, val_labels)),
                               float(val_cost))

        return costs, val_costs

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
        confusion_matrix = self.compute_confusion(test_predictions, test_labels)
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
