from abc import ABC, abstractmethod


class TrainIterator(ABC):
    @abstractmethod
    def run(self, epochs, perform_epoch, epoch_print_cost, epoch_callback, evaluate_valset):
        pass
