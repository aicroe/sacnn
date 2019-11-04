from .kim_arch import KimArch
from .kim_1fc_arch import Kim1FcArch
from .kim_pc_arch import KimPcArch
from .kim_1fc_pc_arch import Kim1FcPcArch

def get_arch(hyperparams):
    sentence_length = hyperparams.get('sentence_length', 100)
    word_dimension = hyperparams.get('word_dimension', 300)
    name = hyperparams['name']
    arch = hyperparams['arch']
    filters = hyperparams['filters']
    num_labels = hyperparams['num_labels']

    # Restored models don't need it
    learning_rate = hyperparams.get('learning_rate', 0)
    keep_prob = hyperparams.get('keep_prob', 0)

    if arch == 'kim':
        filter_height, num_filters = filters[0]
        return KimArch(
            name,
            sentence_length,
            word_dimension,
            filter_height,
            num_filters,
            num_labels,
            learning_rate,
            keep_prob=keep_prob,
        )
    elif arch == 'kim1fc':
        filter_height, num_filters = filters[0]
        hidden_units = hyperparams['hidden_units']
        return Kim1FcArch(
            name,
            sentence_length,
            word_dimension,
            filter_height,
            num_filters,
            hidden_units,
            num_labels,
            learning_rate,
            keep_prob=keep_prob,
        )
    elif arch == 'kimpc':
        return KimPcArch(
            name,
            sentence_length,
            word_dimension,
            filters,
            num_labels,
            learning_rate,
            keep_prob=keep_prob,
        )
    elif arch == 'kim1fcpc':
        hidden_units = hyperparams['hidden_units']
        return Kim1FcPcArch(
            name,
            sentence_length,
            word_dimension,
            filters,
            hidden_units,
            num_labels,
            learning_rate,
            keep_prob=keep_prob,
        )
