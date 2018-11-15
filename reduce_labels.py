import numpy as np
from os import sys
from lib.data_saver import DataSaver


def reduce_set(dataset, labels):
    assert dataset.shape[0] == labels.shape[0]

    positives = labels[:, 0] == 1
    neutrals = labels[:, 2] == 1
    negatives = labels[:, 4] == 1

    new_labels = np.concatenate((
        labels[positives][:, 0:3],
        labels[neutrals][:, 1:4],
        labels[negatives][:, 2:5],
    ), axis=0)

    new_dataset = np.concatenate((
        dataset[positives],
        dataset[neutrals],
        dataset[negatives],
    ), axis=0)

    new_indices = np.random.permutation(new_dataset.shape[0])
    np.take(new_labels, new_indices, axis=0, out=new_labels)
    np.take(new_dataset, new_indices, axis=0, out=new_dataset)

    assert new_dataset.shape[0] == new_labels.shape[0]
    return new_dataset, new_labels


if __name__ == '__main__':

    (train_dataset,
     train_labels,
     val_dataset,
     val_labels,
     test_dataset,
     test_labels) = DataSaver.load_data()

    new_train_dataset, new_train_labels = reduce_set(train_dataset, train_labels)
    new_val_dataset, new_val_labels = reduce_set(val_dataset, val_labels)
    new_test_dataset, new_test_labels = reduce_set(test_dataset, test_labels)

    print('train dataset shape: ({}, {}, {}, {})'.format(*new_train_dataset.shape))
    print('train labels shape: ({}, {})'.format(*new_train_labels.shape))
    print('val dataset shape: ({}, {}, {}, {})'.format(*new_val_dataset.shape))
    print('val labels shape: ({}, {})'.format(*new_val_labels.shape))
    print('test dataset shape: ({}, {}, {}, {})'.format(*new_test_dataset.shape))
    print('test labels shape: ({}, {})'.format(*new_test_labels.shape))

    print('Data ratings : ' + str(list(range(1, 4))))
    print('------ train : ' + str(np.sum(new_train_labels, axis=0)))
    print('------ val   : ' + str(np.sum(new_val_labels, axis=0)))
    print('------ test  : ' + str(np.sum(new_test_labels, axis=0)))

    DataSaver.save_reduced_data(
        new_train_dataset,
        new_train_labels,
        new_val_dataset,
        new_val_labels,
        new_test_dataset,
        new_test_labels)
