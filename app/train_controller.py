from lib.sacnn_validator import SACNNValidator
from lib.data_saver import DataSaver
from .app_state import AppState
from .app_controller import AppController
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class InstanceState(object):
    def __init__(self):
        self.state = 'initializing'
        self.epoch = 0
        self.costs = None
        self.val_costs = None
        self.test_cost = None
        self.test_accuracy = None
        self.confusion_matrix = None

    def to_dict(self):
        if self.state == 'initializing' or self.state == 'training':
            return {
                'state': self.state,
                'epoch': self.epoch
            }
        else:
            print(type(self.state), type(self.epoch), type(self.costs), type(self.val_costs), type(
                self.test_cost), type(self.test_accuracy), type(self.confusion_matrix))
            return {
                'state': self.state,
                'epoch': self.epoch,
                'costs': self.costs,
                'val_costs': self.val_costs,
                'test_cost': self.test_cost,
                'test_accuracy': self.test_accuracy,
                'confusion_matrix': self.confusion_matrix
            }


def create_train_callback(instance_state, epochs):
    def train_callback(epoch,
                       minibatch_accuarcy,
                       minibatch_cost,
                       val_accuracy=None,
                       val_cost=None):
        print(epoch, minibatch_accuarcy, minibatch_cost, val_accuracy, val_cost)
        instance_state.state = 'training'
        instance_state.epoch = epoch
    return train_callback


class TrainController(AppController):

    def __init__(self, app_state):
        """
        :param AppState app_state:
        """
        self.app_state = app_state
        self.training_instances = {}

    def get_training_state(self, name):
        return self.training_instances[name]

    def train_instance(self, hyperparams):
        """
        :param dict hyperparams:
        """

        SACNNValidator.validate(['arch', 'name', 'epochs'], hyperparams)

        hyperparams['sentence_length'] = self.sentence_length
        hyperparams['word_dimension'] = self.word_dimension
        hyperparams['filters_size'] = self.filters_size

        builder = self.builder[hyperparams['arch']]
        name = hyperparams['name']
        self.app_state.record_instance(name,
                                       hyperparams.get('hidden_units', 0),
                                       hyperparams['num_labels'],
                                       hyperparams['arch'])
        self.training_instances[name] = InstanceState()
        epochs = hyperparams['epochs']
        callback = create_train_callback(self.training_instances[name], int(epochs))

        try:
            (_,
             costs,
             val_costs,
             test_cost,
             test_accuracy,
             confusion_matrix) = builder.train(hyperparams, callback)
        except Exception:
            self.app_state.remove_instance(hyperparams['name'])
            del self.training_instances[name]
            return

        self.training_instances[name].state = 'finished'
        self.training_instances[name].costs = costs
        # self.training_instances[name].val_costs = val_costs
        # self.training_instances[name].test_cost = test_cost.tolist()
        # self.training_instances[name].test_accuracy = test_accuracy.tolist()
        # self.training_instances[name].confusion_matrix = confusion_matrix.tolist()

        epoch_print_cost = int(hyperparams['epoch_print_cost'])
        learning_rate = float(hyperparams['learning_rate'])

        plots = plt.plot(
            [x for x in range(len(costs))], costs, 'C0',
            [x * epoch_print_cost for x in range((len(val_costs)))], val_costs, 'r')
        plt.ylabel('Costo')
        plt.xlabel('Iteraciones')
        plt.legend(plots, ('Entrenamiento', 'Validación'))
        plt.title('Tasa de aprendizaje = %.3f' % learning_rate)
        plt.savefig(str(DataSaver.get_app_dir().joinpath('train-cost-%s.png' % hyperparams['name'])))
        print('Test Accuracy:\n', test_accuracy)
        print('Test Cost:', test_cost)
        print('Confusion Matrix:\n', confusion_matrix)
