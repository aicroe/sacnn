from lib.classifier_factory import ClassifierFactory
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
    test_dataset, test_labels = DataSaver.load_data(just_test_data=True)
else:
    test_dataset, test_labels = DataSaver.load_reduced_data(just_test_data=True)
    
_, sentence_length, word_dimension, channels = test_dataset.shape
_, num_labels = test_labels.shape
filters_size = [(3, 96), (5, 96), (7, 64)]
hidden_units = 64

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

print('parameters restored', model.restore())

test_cost, test_accuracy, confusion_matrix = model.test(test_dataset, test_labels)
print('accuarcy over test set: %f' % test_accuracy)
print('cost over test set: %f' % test_cost)
print('confusion matrix:\n', confusion_matrix)
print('confision matrix accuracy:', SACNN.confusion_matrix_accuracy(confusion_matrix))
