from lib.data_saver import DataSaver
from lib.helpers.confusion_matrix import ConfusionMatrix
from lib.sacnn_builders.evolved_sacnn_builder import EvolvedCANNBuilder
from lib.sacnn_builders.kim_sacnn_builder import KimSCANNBuilder
from hyperparams_list import hyperparams_list

confusion_matrix_helper = ConfusionMatrix.get_instance()

load_data = {
    3: DataSaver.load_reduced_data,
    5: DataSaver.load_data
}

builders = {
    'kim': KimSCANNBuilder.get_instance(),
    'evolved': EvolvedCANNBuilder.get_instance()
}

for hyperparams in hyperparams_list:
    name = hyperparams['name']
    arch = hyperparams['arch']
    num_labels = hyperparams['num_labels']

    print('------ Restore Model: %s ------' % name)
    sacnn = builders[arch].restore(hyperparams)
    test_dataset, test_labels = load_data[num_labels](True)
    test_cost, test_accuracy, confusion_matrix = sacnn.test(test_dataset, test_labels)
    print('Accuarcy over test set: %f' % test_accuracy)
    print('Cost over test set: %f' % test_cost)
    print('Confusion matrix:\n', confusion_matrix)
    print('Confision matrix accuracy:', confusion_matrix_helper.accuracy(confusion_matrix))
