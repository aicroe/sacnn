import math
from lib.helpers.accuracy import Accuracy
from .train_iterator import TrainIterator


class EarlyStopIterator(TrainIterator):
    def __init__(self, patience=5, accuracy=Accuracy.get_instance()):
        self.patience = patience
        self.accuracy = accuracy

    def run(self, epochs, perform_epoch, epoch_print_cost, epoch_callback, evaluate_valset):
        epoch = 0
        costs = []
        val_costs = []
        current_val_cost = math.inf
        iterations_failing = 0
        epochs_between_eval = epoch_print_cost

        while epoch < epochs and iterations_failing < self.patience:
            for _ in range(epochs_between_eval):
                minibatch_cost, minibatch_accuarcy = perform_epoch()
                costs.append(minibatch_cost)

            epoch += epochs_between_eval
            val_predictions, val_cost, val_labels = evaluate_valset()
            val_costs.append(val_cost)
            epoch_callback(int(epoch),
                           float(minibatch_accuarcy),
                           float(minibatch_cost),
                           float(self.accuracy(val_predictions, val_labels)),
                           float(val_cost))
            if val_cost < current_val_cost:
                iterations_failing = 0
                current_val_cost = val_cost
            else:
                iterations_failing += 1

        if iterations_failing >= self.patience and epoch < epochs:
            print('Early stop performed at epoch %d' % epoch)
        return epoch, costs, val_costs
