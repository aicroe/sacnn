from abc import ABC


class SACNNValidator(ABC):

    def __init__(self):
        super().__init__()

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
