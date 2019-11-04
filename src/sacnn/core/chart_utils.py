import matplotlib.pyplot as plt

from .fs_utils import get_app_dir

def draw_learning_curve(
        name,
        costs,
        val_costs,
        validation_gap,
        learning_rate):
    plt.clf()
    plots = plt.plot(
        [x for x in range(len(costs))], costs, 'C0',
        [x * validation_gap for x in range((len(val_costs)))], val_costs, 'r')
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.legend(plots, ('Train set', 'Validation set'))
    plt.title('Learning Rate = %.3f' % learning_rate)

    chart_name = '%s-learning-curve.png' % name
    chart_fig_save_path = str(get_app_dir().joinpath(chart_name))
    plt.savefig(chart_fig_save_path)

    return chart_fig_save_path

def draw_accuracies_chart(
        name,
        accuracies,
        val_accuracies,
        validation_gap):
    plt.clf()
    iterations = len(val_accuracies)
    plots = plt.plot(
        [x for x in range(iterations)], accuracies, 'g',
        [x * validation_gap for x in range(iterations)], val_accuracies, 'm')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(plots, ('Train set', 'Validation set'))

    chart_name = '%s-accuracies.png' % name
    chart_fig_save_path = str(get_app_dir().joinpath(chart_name))
    plt.savefig(chart_fig_save_path)

    return chart_fig_save_path
