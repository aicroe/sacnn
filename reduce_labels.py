import numpy as np
from model import DataManager

if __name__ == '__main__':
    NEW_NUM_LABELS = 3
    (train_dataset,
    train_labels,
    val_dataset,
    val_labels,
    test_dataset,
    test_labels) = DataManager.load_train_data()

    new_train_labels = np.zeros((train_labels.shape[0], NEW_NUM_LABELS))
    new_val_labels = np.zeros((val_labels.shape[0], NEW_NUM_LABELS))
    new_test_labels = np.zeros((test_labels.shape[0], NEW_NUM_LABELS))

    new_train_labels[:, 0] = train_labels[:, 0] + train_labels[:, 1]
    new_train_labels[:, 1] = train_labels[:, 2]
    new_train_labels[:, 2] = train_labels[:, 3] + train_labels[:, 4]

    new_val_labels[:, 0] = val_labels[:, 0] + val_labels[:, 1]
    new_val_labels[:, 1] = val_labels[:, 2]
    new_val_labels[:, 2] = val_labels[:, 3] + val_labels[:, 4]

    new_test_labels[:, 0] = test_labels[:, 0] + test_labels[:, 1]
    new_test_labels[:, 1] = test_labels[:, 2]
    new_test_labels[:, 2] = test_labels[:, 3] + test_labels[:, 4]


    print('train dataset shape: ({}, {}, {}, {})'.format(*train_dataset.shape))
    print('train labels shape: ({}, {})'.format(*new_train_labels.shape))
    print('val dataset shape: ({}, {}, {}, {})'.format(*val_dataset.shape))
    print('val labels shape: ({}, {})'.format(*new_val_labels.shape))
    print('test dataset shape: ({}, {}, {}, {})'.format(*test_dataset.shape))
    print('test labels shape: ({}, {})'.format(*new_test_labels.shape))
    ## TODO: display the next with graphics
    print('Data ratings : ' + str(list(range(1, NEW_NUM_LABELS + 1))))
    print('------ train : ' + str(np.sum(new_train_labels, axis=0)))
    print('------ val   : ' + str(np.sum(new_val_labels, axis=0)))
    print('------ test  : ' + str(np.sum(new_test_labels, axis=0)))

    DataManager.save_data(
        train_dataset,
        new_train_labels,
        val_dataset,
        new_val_labels,
        test_dataset,
        new_test_labels)
