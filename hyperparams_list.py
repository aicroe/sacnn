hyperparams_list = [
    {
        'name': 'kim01',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(3, 96), (5, 96), (7, 64)],
        'num_labels': 5,
        'learning_rate': 0.009,
        'epochs': 5,
        'epoch_print_cost': 1,
        # 'epochs': 201,
        # 'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'name': 'evolved01',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(3, 96), (5, 96), (7, 64)],
        'hidden_units': 64,
        'num_labels': 5,
        'learning_rate': 0.009,
        'epochs': 5,
        'epoch_print_cost': 1,
        # 'epochs': 201,
        # 'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    }
]
