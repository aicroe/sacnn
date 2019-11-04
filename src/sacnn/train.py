import sys
import json
import matplotlib
matplotlib.use('Agg')

from mlscratch.train_watcher import TrainWatcher
from mlscratch.measurer import ProbsMeasurer

from sacnn.core import SacnnModel, get_arch, get_trainer, ConfusionMatrixMeasurer, compute_labels_accuracy
from sacnn.core.fs_utils import load_np_tensor, get_app_dir
from sacnn.core.chart_utils import draw_learning_curve, draw_accuracies_chart


class TrainLogger(TrainWatcher):

    def on_epoch(self, epoch, cost, accuracy):
        print('···· Epoch: %d' % epoch)
        print('epoch accuracy : %f' % accuracy)
        print('epoch cost     : %f' % cost)

    def on_validation_epoch(self, epoch, cost, accuracy):
        print('val accuracy   : %f' % accuracy)
        print('val cost       : %f' % cost)

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


def main():
    hyperparams_list = json.load(sys.stdin)
    summary = [(
        'name',
        'arch',
        'trainer',
        'epochs',
        'learning_rate',
        'parallel_convs',
        'accuracy',
    )]

    for hyperparams in hyperparams_list:
        name = hyperparams['name']
        arch_name = hyperparams['arch']
        trainer_name = hyperparams['trainer']
        num_labels = hyperparams['num_labels']
        epochs = hyperparams['epochs']
        validation_gap = hyperparams['validation_gap']
        minibatch_size = hyperparams['minibatch_size']
        learning_rate = hyperparams['learning_rate']
        parallel_convs = len(hyperparams['filters'])

        model = SacnnModel(get_arch(hyperparams))
        trainer = get_trainer(trainer_name)

        print('··· START training: "%s"' % name)

        (train_dataset,
         train_labels,
         val_dataset,
         val_labels,
         test_dataset,
         test_labels) = get_data(num_labels)

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
            TrainLogger(),
            epochs=epochs,
            validation_gap=validation_gap,
            minibatch_size=minibatch_size,
        )

        draw_learning_curve(name, costs, val_costs, validation_gap, learning_rate)
        draw_accuracies_chart(name, accuracies, val_accuracies, validation_gap)

        test_cost, [test_accuracy, confusion_mat] = model.measure(
            test_dataset,
            test_labels,
            [ProbsMeasurer(), ConfusionMatrixMeasurer()],
        )

        print('Epochs: %f' % actual_epochs)
        print('Test Set Cost: %f' % test_cost)
        print('Test Set Accuracy: %f' % test_accuracy)
        print('Confusion Matrix:\n', confusion_mat)
        print('Labels Accuracy:\n', compute_labels_accuracy(confusion_mat))

        model.save()

        print('··· FINISH training "%s"' % name)

        summary.append((
            name,
            arch_name,
            trainer_name,
            actual_epochs,
            learning_rate,
            parallel_convs,
            test_accuracy,
        ))

    print('··· Summary ···')
    for each in summary:
        print(str.join(',', map(str, each)))
