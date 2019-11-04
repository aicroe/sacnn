from mlscratch.trainer import SimpleTrainer, SgdTrainer, EarlyStopTrainer, SgdEarlyStopTrainer

from .static_patience_decorator import StaticPatienceDecorator

def get_trainer(trainer, patience=5):
    if trainer == 'simple':
        return SimpleTrainer()
    elif trainer == 'sgd':
        return SgdTrainer()
    elif trainer == 'early_stop':
        return StaticPatienceDecorator(EarlyStopTrainer(), patience)
    elif trainer == 'sgd_early_stop':
        return StaticPatienceDecorator(SgdEarlyStopTrainer(), patience)
