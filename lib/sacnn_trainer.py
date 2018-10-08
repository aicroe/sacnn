from .data_saver import DataSaver
from .hyperparams import Hyperparams
from abc import ABC


class SACNNTrainer(ABC):

    def __init__(self):
        super().__init__()
        self.load_data = {
            3: DataSaver.load_reduced_data,
            5: DataSaver.load_data
        }

    def train(self, hyperparams, callback):
        sacnn = self.create(hyperparams)
        (train_dataset,
         train_labels,
         val_dataset,
         val_labels,
         test_dataset,
         test_labels) = self.load_data[int(hyperparams['num_labels'])]()

        self.validate(['learning_rate', 'epochs', 'epoch_print_cost',
                       'minibatch_size', 'keep_prob'], hyperparams)

        epochs = int(hyperparams['epochs'])
        epoch_print_cost = int(hyperparams['epoch_print_cost'])
        if epoch_print_cost > epochs:
            raise BaseException('ilegal_epoch_print_cost')

        hparams = Hyperparams(float(hyperparams['learning_rate']),
                              epochs,
                              int(hyperparams['minibatch_size']),
                              float(hyperparams['keep_prob']))
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
        sacnn.save()

        return sacnn, costs, val_costs, test_cost, test_accuracy, confusion_matrix
