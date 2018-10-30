hyperparams_list = [
    # filters size and number evaluation on evolved arch
    {
        'arch': 'evolved',
        'name': 'evolved_3_h16',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(3, 64)],
        'hidden_units': 16,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_3_h32',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(3, 64)],
        'hidden_units': 32,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_3_h64',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(3, 64)],
        'hidden_units': 64,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_3_h128',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(3, 64)],
        'hidden_units': 128,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_4_h16',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(4, 64)],
        'hidden_units': 16,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_4_h32',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(4, 64)],
        'hidden_units': 32,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_4_h64',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(4, 64)],
        'hidden_units': 64,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_4_h128',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(4, 64)],
        'hidden_units': 128,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_5_h16',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(5, 128)],
        'hidden_units': 16,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_5_h32',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(5, 128)],
        'hidden_units': 32,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_5_h64',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(5, 128)],
        'hidden_units': 64,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_5_h128',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(5, 128)],
        'hidden_units': 128,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_6_h16',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(6, 96)],
        'hidden_units': 16,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_6_h32',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(6, 96)],
        'hidden_units': 32,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_6_h64',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(6, 96)],
        'hidden_units': 64,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_6_h128',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(6, 96)],
        'hidden_units': 128,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_7_h16',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(7, 64)],
        'hidden_units': 16,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_7_h32',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(7, 64)],
        'hidden_units': 32,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_7_h64',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(7, 64)],
        'hidden_units': 64,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_7_h128',
        'iterator': 'simple',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(7, 64)],
        'hidden_units': 128,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 201,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_3_h16_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(3, 128)],
        'hidden_units': 16,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_3_h32_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(3, 128)],
        'hidden_units': 32,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_3_h64_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(3, 128)],
        'hidden_units': 64,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_3_h128_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(3, 128)],
        'hidden_units': 128,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_4_h16_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(4, 128)],
        'hidden_units': 16,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_4_h32_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(4, 128)],
        'hidden_units': 32,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_4_h64_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(4, 128)],
        'hidden_units': 64,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_4_h128_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(4, 128)],
        'hidden_units': 128,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_5_h16_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(5, 64)],
        'hidden_units': 16,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_5_h32_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(5, 64)],
        'hidden_units': 32,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_5_h64_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(5, 64)],
        'hidden_units': 64,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_5_h128_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(5, 64)],
        'hidden_units': 128,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_6_h16_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(6, 128)],
        'hidden_units': 16,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_6_h32_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(6, 128)],
        'hidden_units': 32,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_6_h64_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(6, 128)],
        'hidden_units': 64,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_6_h128_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(6, 128)],
        'hidden_units': 128,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_7_h16_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(7, 128)],
        'hidden_units': 16,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_7_h32_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(7, 128)],
        'hidden_units': 32,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_7_h64_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(7, 128)],
        'hidden_units': 64,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    },
    {
        'arch': 'evolved',
        'name': 'evolved_7_h128_early',
        'iterator': 'early_stop',
        'sentence_length': 100,
        'word_dimension': 300,
        'filters_size': [(7, 128)],
        'hidden_units': 128,
        'num_labels': 3,
        'learning_rate': 0.009,
        'epochs': 301,
        'epoch_print_cost': 5,
        'minibatch_size': 32,
        'keep_prob': 0.5
    }
]
