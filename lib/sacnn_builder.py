from .data_saver import DataSaver
from .hyperparams import Hyperparams
from abc import abstractmethod, ABC


class SACNNBuilder(ABC):
    channels = 1

    def __init__(self, hyperkeys):
        self.hyperkeys = hyperkeys
        self.load_data = {
            3: DataSaver.load_reduced_data,
            5: DataSaver.load_data
        }

    @staticmethod
    def validate(keys, dictionary):
        for key in keys:
            if dictionary[key] is None:
                raise BaseException('%s_cannot_be_none' % key)

    def validate_hparams(self, hyperparams):
        self.validate(self.hyperkeys, hyperparams)
        if 'num_labels' in self.hyperkeys:
            if hyperparams['num_labels'] != 3 and hyperparams['num_labels'] != 5:
                raise BaseException('num_labels_must_be_3_or_5')

    @abstractmethod
    def create(self, hyperparams):
        pass

    def restore(self, hyperparams):
        sacnn = self.create(hyperparams)
        sacnn.restore()
        return sacnn

    def train(self, hyperparams, trainhparams, callback):
        sacnn = self.create(hyperparams)
        (train_dataset,
         train_labels,
         val_dataset,
         val_labels,
         test_dataset,
         test_labels) = self.load_data[hyperparams['num_labels']]()
        self.validate(['learning_rate', 'epochs', 'epoch_print_cost',
                       'minibatch_size', 'keep_prob'], trainhparams)
        epoch_print_cost = trainhparams['epoch_print_cost']
        hparams = Hyperparams(trainhparams['learning_rate'],
                              trainhparams['epochs'],
                              trainhparams['minibatch_size'],
                              trainhparams['keep_prob'])
        costs, val_costs = sacnn.train(
            train_dataset,
            train_labels,
            hparams,
            epoch_print_cost,
            callback,
            val_dataset,
            val_labels)

        test_cost, test_accuracy, confusion_matrix = sacnn.test(
            test_dataset, test_labels)

        return sacnn, costs, val_costs, test_cost, test_accuracy, confusion_matrix
