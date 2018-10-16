from lib.helpers.accuracy import Accuracy
from lib.singleton_decorator import singleton
from .train_iterator import TrainIterator


@singleton()
class SimpleIterator(TrainIterator):
    def __init__(self, accuracy=Accuracy.get_instance()):
        super().__init__()
        self.accuracy = accuracy

    def run(self, epochs, perform_epoch, epoch_print_cost, epoch_callback, evaluate_valset):
        costs = []
        val_costs = []
        for epoch in range(epochs):
            minibatch_cost, minibatch_accuarcy = perform_epoch()
            costs.append(minibatch_cost)

            if epoch_print_cost > 0 and epoch % epoch_print_cost == 0:
                val_predictions, val_cost, val_labels = evaluate_valset()
                val_costs.append(val_cost)
                epoch_callback(int(epoch),
                               float(minibatch_accuarcy),
                               float(minibatch_cost),
                               float(self.accuracy(val_predictions, val_labels)),
                               float(val_cost))
        return epochs, costs, val_costs
