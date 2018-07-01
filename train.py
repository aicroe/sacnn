import matplotlib.pyplot as plt
from lib.classifier_factory import ClassifierFactory
from lib.hyperparams import Hyperparams
from lib.sacnn import SACNN
from lib.data_saver import DataSaver
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--arch',  default='evolved', type=str)
parser.add_argument('--labels', default=5, type=int)
args = parser.parse_args()

if (args.arch != 'base' and args.arch != 'evolved') or (args.labels != 3 and args.labels != 5):
    raise BaseException('[wrong arguments]')

if args.labels == 5:
    (train_dataset,
    train_labels,
    val_dataset,
    val_labels,
    test_dataset,
    test_labels) = DataSaver.load_data()
else:
    (train_dataset,
    train_labels,
    val_dataset,
    val_labels,
    test_dataset,
    test_labels) = DataSaver.load_reduced_data()

_, sentence_length, word_dimension, channels = train_dataset.shape
_, num_labels = train_labels.shape
filters_size = [(3, 96), (5, 96), (7, 64)]
hidden_units = 64

hparams = Hyperparams(learning_rate=0.009,
                        epochs=201,
                        minibatch_size=32,
                        keep_prob=0.5)

name = '%s-c%d' % (args.arch, args.labels)
if args.arch == 'evolved':
    model = ClassifierFactory.evolved_model(name,
                                            word_dimension,
                                            sentence_length,
                                            channels,
                                            filters_size,
                                            hidden_units,
                                            num_labels)
else:
    model = ClassifierFactory.base_model(name,
                                         word_dimension,
                                         sentence_length,
                                         channels,
                                         filters_size,
                                         num_labels)

epoch_print_cost = 5
def epoch_callback(epoch,
                    minibatch_accuarcy,
                    minibatch_cost,
                    val_accuracy=None,
                    val_cost=None):
    if val_accuracy is not None and val_cost is not None:
        print('--------- epoch %d ----------' % epoch)
        print('minibatch accuracy: %f' % minibatch_accuarcy)
        print('minibatch cost    : %f' % minibatch_cost)
        print('val accuracy      : %f' % val_accuracy)
        print('val cost          : %f' % val_cost)

costs, val_costs = model.train(train_dataset,
            train_labels,
            hparams,
            epoch_print_cost,
            epoch_callback,
            val_dataset,
            val_labels)

plots = plt.plot(
    [x for x in range(len(costs))], costs, 'C0',
    [x * epoch_print_cost for x in range((len(val_costs)))], val_costs, 'r')
plt.ylabel('Costo')
plt.xlabel('Iteraciones')
plt.legend(plots, ('Entrenamiento', 'Validación'))
plt.title('Tasa de aprendizaje = %.3f' % hparams.learning_rate)
plt.savefig(str(DataSaver.get_app_dir().joinpath('train-cost-%s.png' % name)))

print('------FINISH training-------')
test_cost, test_accuracy, confusion_matrix = model.test(test_dataset, test_labels)
print('accuarcy over test set: %f' % test_accuracy)
print('cost over test set: %f' % test_cost)
print('confusion matrix:\n', confusion_matrix)
print('confision matrix accuracy:', SACNN.confusion_matrix_accuracy(confusion_matrix))

print('re')
test_cost, test_accuracy, confusion_matrix = model.test(test_dataset, test_labels)
print('accuarcy over test set: %f' % test_accuracy)
print('cost over test set: %f' % test_cost)
print('confusion matrix:\n', confusion_matrix)
print('confision matrix accuracy:', SACNN.confusion_matrix_accuracy(confusion_matrix))

print('parameters saved', model.save())
