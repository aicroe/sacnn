from model import Parameters, HyperParameters, SACNNBase, DataManager
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / predictions.shape[0])


def generate_random_minibatches(inputs, outputs, minibatch_size):
    assert inputs.shape[0] == outputs.shape[0]
    batch_size = inputs.shape[0]
    permutation = np.random.permutation(batch_size)
    inputs_shuffled = inputs[permutation]
    outputs_shuffeld = outputs[permutation]
    num_batches = math.ceil(batch_size / minibatch_size)
    return zip(np.array_split(inputs_shuffled, num_batches), np.array_split(outputs_shuffeld, num_batches))


class SACNN(object):
    def __init__(self,
                 sentence_length,
                 word_dimension,
                 channels,
                 num_labels,
                 filter_size,
                 depth,
                 learning_rate=0.009,
                 keep_prob=0.5):

        # Define placeholders
        self.tf_dataset = tf.placeholder(tf.float32, [None, sentence_length, word_dimension, channels])
        self.tf_labels = tf.placeholder(tf.float32, [None, num_labels])
        self.tf_keep_prob = tf.placeholder(tf.float32)

        # Define the parameters
        layer1_filters = tf.Variable(tf.truncated_normal([filter_size, word_dimension, channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal([depth, num_labels], stddev=0.1))
        layer2_biases = tf.Variable(tf.zeros([num_labels]))

        self._parameters = Parameters(
            layer1_filters,
            layer1_biases,
            layer2_weights,
            layer2_biases)

        self._hparameters = HyperParameters(
            filter_size,
            word_dimension,
            learning_rate)

        self._keep_prob = keep_prob
        
        self._model = SACNNBase(
            self._parameters,
            self._hparameters,
            self.tf_dataset,
            self.tf_labels,
            self.tf_keep_prob)
        self._prediction = self._model.prediction
        self._optimizer = self._model.optimizer
        self._cost = self._model.cost

    def get_prediction(self):
        return self._prediction

    def get_parameters(self):
        return self._parameters

    def train(self, 
              session,
              train_dataset,
              train_labels,
              val_dataset,
              val_labels,
              minibatch_size,
              epochs,
              epoch_print_cost=0):
        costs = []
        val_costs = []
        num_minibatches = math.ceil(train_dataset.shape[0] / minibatch_size)
        for epoch in range(epochs):
            minibatch_cost = 0.
            minibatch_accuarcy = 0.
            minibatches = generate_random_minibatches(train_dataset, train_labels, minibatch_size)
            for (minibatch_dataset, minibatch_labels) in minibatches:
                train_dict = {
                    self.tf_dataset: minibatch_dataset,
                    self.tf_labels: minibatch_labels,
                    self.tf_keep_prob: self._keep_prob
                }
                tensors = [self._prediction, self._optimizer, self._cost]
                train_predictions, _, current_cost = session.run(tensors, feed_dict=train_dict)
                minibatch_cost += current_cost / num_minibatches
                minibatch_accuarcy += accuracy(train_predictions, minibatch_labels) / num_minibatches
            #if (minibatch_cost < 10):
            costs.append(minibatch_cost)
            if epoch_print_cost > 0 and epoch % epoch_print_cost == 0:
                print('--------epoch: %d-------' % epoch)
                print('cost over training set: %f' % minibatch_cost)
                print('accuarcy over training set: %f' % minibatch_accuarcy)
                val_dict = {
                    self.tf_dataset: val_dataset,
                    self.tf_labels: val_labels,
                    self.tf_keep_prob: 1
                }
                val_predictions = self._prediction.eval(session=session, feed_dict=val_dict)
                val_cost = self._cost.eval(session=session, feed_dict=val_dict)
                val_costs.append(val_cost)
                print('cost over validation set: %f' % val_cost)
                print('accuarcy over validation set: %f' % accuracy(val_predictions, val_labels))
        return costs, val_costs


def _main():
    (train_dataset,
    train_labels,
    val_dataset,
    val_labels,
    test_dataset,
    test_labels) = DataManager.load_train_data()
    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(0)
    session = tf.Session()

    _, sentence_length, word_dimension, channels = train_dataset.shape
    raw_labels = [1, 2, 3, 4, 5]

    filter_size = 3
    depth = 64 #100
    keep_prob = 0.5
    learning_rate = 0.009

    model = SACNN(
        sentence_length,
        word_dimension,
        channels,
        len(raw_labels),
        filter_size,
        depth,
        learning_rate,
        keep_prob)

    prediction = model.get_prediction()
    tf_dataset = model.tf_dataset
    tf_labels = model.tf_labels
    tf_keep_prob = model.tf_keep_prob

    init = tf.global_variables_initializer()
    session.run(init)

    epochs = 201
    minibatch_size = 32
    epoch_print_cost = 1

    costs, val_costs = model.train(
        session,
        train_dataset,
        train_labels,
        val_dataset,
        val_labels,
        minibatch_size,
        epochs,
        epoch_print_cost)

    plt.plot(
        [x for x in range(len(costs))], costs, 'b', 
        [x * epoch_print_cost for x in range((len(val_costs)))], val_costs, 'r')
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Learning rate = %d' % learning_rate)
    plt.savefig('train-cost.png')

    print('-------training FINISHED---------')
    test_dict = {
        tf_dataset: test_dataset,
        tf_labels: test_labels,
        tf_keep_prob: 1
    }
    test_predictions = prediction.eval(session=session, feed_dict=test_dict)
    print('accuarcy over test set: %f' % accuracy(test_predictions, test_labels))
    test_predictions = prediction.eval(session=session, feed_dict=test_dict)
    print('accuarcy over test set: %f' % accuracy(test_predictions, test_labels))

    parameters = model.get_parameters()
    DataManager.save_parameters(
        parameters.layer1_filters.eval(session=session),
        parameters.layer1_biases.eval(session=session),
        parameters.layer2_weights.eval(session=session),
        parameters.layer2_biases.eval(session=session))
    print('parameters saved at data/')

    session.close()


if __name__ == '__main__':
    _main()
