import sys
import json

from mlscratch.measurer import ProbsMeasurer, Measurer

from sacnn.core import SacnnModel, get_arch, compute_labels_accuracy, ConfusionMatrixMeasurer
from sacnn.core.fs_utils import load_np_tensor

def get_test_data(num_labels):
    load_folder = 'data_reduced' if num_labels == 3 else 'data'
    test_dataset = load_np_tensor('test_dataset', load_folder)
    test_labels = load_np_tensor('test_labels', load_folder)
    return test_dataset, test_labels

def main():
    hyperparams_list = json.load(sys.stdin)
    for hyperparams in hyperparams_list:
        name = hyperparams['name']
        num_labels = hyperparams['num_labels']

        print('··· Restore Model: "%s"' % name)
        model = SacnnModel(get_arch(hyperparams))
        model.restore()

        test_dataset, test_labels = get_test_data(num_labels)
        cost, [test_accuracy, confusion_mat] = model.measure(
            test_dataset,
            test_labels,
            [ProbsMeasurer(), ConfusionMatrixMeasurer()],
        )
        print('Cost: %f' % cost)
        print('Test Accuracy: %f' % test_accuracy)
        print('Confusion Matrix:\n', confusion_mat)
        print('Labels Accuracy:\n', compute_labels_accuracy(confusion_mat))
