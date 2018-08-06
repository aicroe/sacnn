import math
import numpy as np
import pandas as pd
from lib.data_saver import DataSaver
from lib.data_processor import DataProcessor
from lib.word_embedding import WordEmbedding


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

    comments_frame = pd.concat([
        comments_rating_1,
        comments_rating_2,
        comments_rating_3,
        comments_rating_4,
        comments_rating_5], ignore_index=True)
    return comments_frame


def _main():
    sentence_length = 100
    channels = 1
    raw_labels = [1, 2, 3, 4, 5]

    print('Loading embedding')
    embedding, _ = WordEmbedding.get_instance()
    print('Loading comments')
    comments_frame = load_comments(str(DataSaver.prepare_dir('raw').joinpath('comments.csv')))
    samples_count, _ = comments_frame.shape

    np.random.seed(0)
    comments_frame = comments_frame.iloc[np.random.permutation(samples_count)].reset_index(drop=True)

    print('Processing comments')
    print(comments_frame['fullContent'].shape)
    data_processor = DataProcessor(embedding.wv, sentence_length, channels)
    samples = data_processor.process(comments_frame['fullContent'])
    onehot_labels = DataProcessor.create_1hot_vectors(comments_frame['rating'], raw_labels)

    # Preparation
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
    
    print('Data ratings : ' + str(raw_labels))
    print('------ train : ' + str(np.sum(train_labels, axis=0)))
    print('------ val   : ' + str(np.sum(val_labels, axis=0)))
    print('------ test  : ' + str(np.sum(test_labels, axis=0)))

    # Save
    DataSaver.save_data(
        train_dataset,
        train_labels,
        val_dataset,
        val_labels,
        test_dataset,
        test_labels)
    print('Saved train, val, test data')


if __name__ == '__main__':
    _main()
