import math
import numpy as np
import pandas as pd

from sacnn.core.we import get_word_to_vector
from sacnn.core.fs_utils import prepare_dir, save_np_tensor
from sacnn.core.preprocessing import process_sentences, create_1hot_vectors

def load_comments(comments_path):
    AMOUNT_EDGE = 400
    types = {
        'rating': np.int32,
        'fullContent': np.str
    }
    comments_frame = pd.read_csv(
        comments_path,
        sep=',',
        header=0,
        encoding='utf-8',
        usecols=list(types.keys()),
        dtype=types)

    comments_rating_1 = comments_frame[comments_frame['rating'] == 1][0:AMOUNT_EDGE]
    comments_rating_2 = comments_frame[comments_frame['rating'] == 2][0:AMOUNT_EDGE]
    comments_rating_3 = comments_frame[comments_frame['rating'] == 3][0:AMOUNT_EDGE]
    comments_rating_4 = comments_frame[comments_frame['rating'] == 4][0:AMOUNT_EDGE]
    comments_rating_5 = comments_frame[comments_frame['rating'] == 5][0:AMOUNT_EDGE]

    print('comments rating=1: %d' % comments_rating_1.shape[0])
    print('comments rating=2: %d' % comments_rating_2.shape[0])
    print('comments rating=3: %d' % comments_rating_3.shape[0])
    print('comments rating=4: %d' % comments_rating_4.shape[0])
    print('comments rating=5: %d' % comments_rating_5.shape[0])

    comments_frame = pd.concat(
        [
            comments_rating_1,
            comments_rating_2,
            comments_rating_3,
            comments_rating_4,
            comments_rating_5,
        ],
        ignore_index=True,
    )
    return comments_frame

def main():
    sentence_length = 100
    raw_labels = [1, 2, 3, 4, 5]

    print('Loading word embedding...')
    word_to_vector, word_dimension = get_word_to_vector(limit=None)

    print('Loading comments...')
    comments_path = str(prepare_dir('raw').joinpath('comments.csv'))
    comments_frame = load_comments(comments_path)
    samples_count, _ = comments_frame.shape

    np.random.seed(0)
    permutation = np.random.permutation(samples_count)
    comments_frame = comments_frame.iloc[permutation].reset_index(drop=True)

    print('Processing comments...')
    print(comments_frame['fullContent'].shape)
    samples = process_sentences(
        comments_frame['fullContent'],
        sentence_length,
        word_dimension,
        word_to_vector,
    )
    onehot_labels = create_1hot_vectors(comments_frame['rating'], raw_labels)

    # Split data
    assert samples.shape[0] == onehot_labels.shape[0]
    eighty_percent = math.floor(samples.shape[0] * 0.80)
    ten_percent = math.floor(samples.shape[0] * 0.10)
    train_dataset = samples[:eighty_percent]
    train_labels = onehot_labels[:eighty_percent]
    val_dataset = samples[eighty_percent:eighty_percent + ten_percent]
    val_labels = onehot_labels[eighty_percent:eighty_percent + ten_percent]
    test_dataset = samples[eighty_percent + ten_percent:]
    test_labels = onehot_labels[eighty_percent + ten_percent:]
    print('train dataset shape: ({}, {}, {}, {})'.format(*train_dataset.shape))
    print('train labels shape: ({}, {})'.format(*train_labels.shape))
    print('val dataset shape: ({}, {}, {}, {})'.format(*val_dataset.shape))
    print('val labels shape: ({}, {})'.format(*val_labels.shape))
    print('test dataset shape: ({}, {}, {}, {})'.format(*test_dataset.shape))
    print('test labels shape: ({}, {})'.format(*test_labels.shape))

    print('Data labels : ' + str(raw_labels))
    print('------ train : ' + str(np.sum(train_labels, axis=0)))
    print('------ val   : ' + str(np.sum(val_labels, axis=0)))
    print('------ test  : ' + str(np.sum(test_labels, axis=0)))

    # Save data
    save_np_tensor('train_dataset', train_dataset, 'data')
    save_np_tensor('train_labels', train_labels, 'data')
    save_np_tensor('val_dataset', val_dataset, 'data')
    save_np_tensor('val_labels', val_labels, 'data')
    save_np_tensor('test_dataset', test_dataset, 'data')
    save_np_tensor('test_labels', test_labels, 'data')
    print('Saved train, val and test data at "data"\'s app folder')
