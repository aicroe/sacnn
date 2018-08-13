from abc import ABC, abstractclassmethod


class SACNNCreator(ABC):
    channels = 1

    def __init__(self, hyperkeys):
        super().__init__()
        self.hyperkeys = hyperkeys

    @abstractclassmethod
    def create(self, hyperparams):
        pass

    def restore(self, hyperparams):
        sacnn = self.create(hyperparams)
        sacnn.restore()
        return sacnn
