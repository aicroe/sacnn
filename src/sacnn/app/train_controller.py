import traceback
import logging
import matplotlib
matplotlib.use('Agg')

from enum import Enum
from mlscratch.train_watcher import TrainWatcher
from mlscratch.measurer import ProbsMeasurer

from sacnn.core import SacnnModel, get_arch, get_trainer, ConfusionMatrixMeasurer, compute_labels_accuracy
from sacnn.core.fs_utils import load_np_tensor
from sacnn.core.chart_utils import draw_learning_curve, draw_accuracies_chart

from .app_state import app_state
from .constraints import FILTERS

logger = logging.getLogger(__name__)

class InstanceStatus(str, Enum):
    Idle = 'IDLE'
    Training = 'TRAINING'
    TrainCompleted = 'TRAIN_COMPLETED'
    TrainFailed = 'TRAIN_FAILED'

class InstanceState(object):
    def __init__(self):
        self.status = InstanceStatus.Idle
        self.epoch = 0
        self.costs = None
        self.accuracies = None
        self.val_costs = None
        self.val_accuracies = None
        self.test_cost = None
        self.test_accuracy = None
        self.learning_curve_path = None

    def to_dict(self):
        if self.status == InstanceStatus.Idle or self.status == InstanceStatus.Training:
            return {
                'status': self.status,
                'epoch': self.epoch
            }
        else:
            return {
                'status': self.status,
                'epoch': self.epoch,
                'costs': [float(cost) for cost in self.costs],
                'accuracies': [float(accuracy) for accuracy in self.accuracies],
                'val_costs': [float(val_cost) for val_cost in self.val_costs],
                'val_accuracies': [float(val_accuracy) for val_accuracy in self.val_accuracies],
                'test_cost': float(self.test_cost),
                'test_accuracy': float(self.test_accuracy),
            }

class InstanceStateUpdater(TrainWatcher):

    def __init__(self, instance_state):
        self._instance_state = instance_state

    def on_epoch(self, epoch, cost, accuracy):
        logger.info('Epoch: %d \t Cost: %f \t Accuracy: %f' % (epoch, cost, accuracy))
        self._instance_state.status = InstanceStatus.Training
        self._instance_state.epoch = epoch

    def on_validation_epoch(self, epoch, cost, accuracy):
        logger.info('Validation Epoch: %d \t Cost: %f \t Accuracy: %f' % (epoch, cost, accuracy))
        self._instance_state.status = InstanceStatus.Training
        self._instance_state.epoch = epoch

def get_data(num_labels):
    load_folder = 'data_reduced' if num_labels == 3 else 'data'
    train_dataset = load_np_tensor('train_dataset', load_folder)
    train_labels = load_np_tensor('train_labels', load_folder)
    val_dataset = load_np_tensor('val_dataset', load_folder)
    val_labels = load_np_tensor('val_labels', load_folder)
    test_dataset = load_np_tensor('test_dataset', load_folder)
    test_labels = load_np_tensor('test_labels', load_folder)

    return (
        train_dataset,
        train_labels,
        val_dataset,
        val_labels,
        test_dataset,
        test_labels,
    )

class TrainController():

    def __init__(self):
        self._instances = { }

    def get_training_instance(self, name):
        return self._instances[name]

    def train_instance(self, hyperparams):
        """
        :param dict hyperparams:
        """

        name = hyperparams['name']
        arch_name = hyperparams['arch']
        trainer_name = hyperparams['trainer']
        num_labels = hyperparams['num_labels']
        hidden_units = hyperparams.get('hidden_units', 0) # Optional param
        epochs = hyperparams['epochs']
        validation_gap = hyperparams['validation_gap']
        minibatch_size = hyperparams['minibatch_size']
        learning_rate = hyperparams['learning_rate']

        hyperparams['filters'] = FILTERS

        model = SacnnModel(get_arch(hyperparams))
        trainer = get_trainer(trainer_name)

        app_state.record_instance(
            name,
            hidden_units,
            num_labels,
            arch_name,
        )

        self._instances[name] = InstanceState()

        (train_dataset,
         train_labels,
         val_dataset,
         val_labels,
         test_dataset,
         test_labels) = get_data(num_labels)

        try:
            (actual_epochs,
             costs,
             accuracies,
             val_epochs,
             val_costs,
             val_accuracies) = model.train(
                train_dataset,
                train_labels,
                val_dataset,
                val_labels,
                trainer,
                ProbsMeasurer(),
                InstanceStateUpdater(self._instances[name]),
                epochs=epochs,
                validation_gap=validation_gap,
                minibatch_size=minibatch_size,
            )
            model.save()
        except:
            traceback.print_exc()
            app_state.remove_instance(hyperparams['name'])
            self._instances[name].status = InstanceStatus.TrainFailed
            return

        self._instances[name].status = InstanceStatus.TrainCompleted
        self._instances[name].costs = costs
        self._instances[name].accuracies = accuracies
        self._instances[name].val_costs = val_costs
        self._instances[name].val_accuracies = val_accuracies

        self._instances[name].learning_curve_path = draw_learning_curve(
            name,
            costs,
            val_costs,
            validation_gap,
            learning_rate,
        )

        test_cost, [test_accuracy, confusion_mat] = model.measure(
            test_dataset,
            test_labels,
            [ProbsMeasurer(), ConfusionMatrixMeasurer()],
        )
        self._instances[name].test_cost = test_cost
        self._instances[name].test_accuracy = test_accuracy

        logger.info('Epochs: %d' % actual_epochs)
        logger.info('Test Set Cost: %f' % float(test_cost))
        logger.info('Test Set Accuracy: %f' % float(test_accuracy))
        logger.info('Confusion Matrix: %s' % str(confusion_mat))
        logger.info('Labels Accuracy: %s' % str(compute_labels_accuracy(confusion_mat)))
