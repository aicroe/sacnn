from mlscratch.trainer import Trainer


class StaticPatienceDecorator(Trainer):

    def __init__(self, early_stop_trainer, patience):
        self._early_stop_trainer = early_stop_trainer
        self._patience = patience

    def train(
            self,
            trainable,
            train_dataset,
            train_labels,
            validation_dataset,
            validation_labels,
            train_watcher,
            **options):

        if options is not None:
            options['patience'] = self._patience
        else:
            options = { 'patience': self._patience }

        return self._early_stop_trainer.train(
            trainable,
            train_dataset,
            train_labels,
            validation_dataset,
            validation_labels,
            train_watcher,
            **options,
        )
