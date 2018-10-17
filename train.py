import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.helpers.confusion_matrix import ConfusionMatrix
from lib.data_saver import DataSaver
from lib.sacnn_builders.evolved_sacnn_builder import EvolvedCANNBuilder
from lib.sacnn_builders.kim_sacnn_builder import KimSCANNBuilder
from lib.train_iterators.early_stop_iterator import EarlyStopIterator
from lib.train_iterators.simple_iterator import SimpleIterator
from hyperparams_list import hyperparams_list

kim_builder = KimSCANNBuilder.get_instance()
evolved_builder = EvolvedCANNBuilder.get_instance()
simple_iterator = SimpleIterator.get_instance()
early_stop_iterator = EarlyStopIterator(3)
confusion_matrix_helper = ConfusionMatrix.get_instance()


def epoch_callback(epoch,
                   minibatch_accuarcy,
                   minibatch_cost,
                   val_accuracy=None,
                   val_cost=None):
    print('--------- epoch: %d ----------' % epoch)
    print('minibatch accuracy: %f' % minibatch_accuarcy)
    print('minibatch cost    : %f' % minibatch_cost)
    print('val accuracy      : %f' % val_accuracy)
    print('val cost          : %f' % val_cost)


models_to_train = [
    # (builder, hyperparams, epoch_callback, train_iterator)
    (kim_builder, hyperparams_list[0], epoch_callback, early_stop_iterator),
    (evolved_builder, hyperparams_list[1], epoch_callback, early_stop_iterator)
]

for (builder, hyperparams, epoch_callback, iterator) in models_to_train:

    name = hyperparams['name']
    epoch_print_cost = hyperparams['epoch_print_cost']
    learning_rate = hyperparams['learning_rate']
    print('------ START training: %s -------' % name)

    (_,
     iterations,
     costs,
     val_costs,
     test_cost,
     test_accuracy,
     confusion_matrix) = builder.train(hyperparams, epoch_callback, iterator)


    plt.clf()
    plots = plt.plot(
        [x for x in range(len(costs))], costs, 'C0',
        [x * epoch_print_cost for x in range((len(val_costs)))], val_costs, 'r')
    plt.ylabel('Costo')
    plt.xlabel('Iteraciones')
    plt.legend(plots, ('Entrenamiento', 'Validación'))
    plt.title('Tasa de aprendizaje = %.3f' % learning_rate)

    learning_curve_path = str(DataSaver.get_app_dir().joinpath('train-cost-%s.png' % name))
    plt.savefig(learning_curve_path)

    print('Iterations:\n', iterations)
    print('Test Accuracy:\n', test_accuracy)
    print('Test Cost:', test_cost)
    print('Confusion Matrix:\n', confusion_matrix)

    print('accuarcy over test set: %f' % test_accuracy)
    print('cost over test set: %f' % test_cost)
    print('confusion matrix:\n', confusion_matrix)
    print('confision matrix accuracy:', confusion_matrix_helper.accuracy(confusion_matrix))
    print('------ FINISH training: %s -------' % name)
