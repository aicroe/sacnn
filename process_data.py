import re
import math
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from model import DataManager


def load_embedding(path):
    # other word2vecs binaries are loaded as:
    # from gensim.models import Word2Vec
    # Word2Vec.load(...)
    return KeyedVectors.load_word2vec_format(path, binary=True)


def clean_str(dirty_word):
    return re.sub(r'[\.,\-"\{\}\[\]\*\^;%\+&°!¡¿?#<>@/\(\)\\=:_~]', '', dirty_word.lower())


def process_comments(
        word_to_vector,
        comments_values,
        samples_count,
        sentence_length,
        word_dimension,
        channels):
    samples = np.empty((samples_count, sentence_length,
                        word_dimension, channels), dtype=np.float32)
    for sample_index in range(samples_count):
        comment = comments_values[sample_index].split()
        sample = np.zeros((sentence_length, word_dimension, 1))
        word_index = 0
        for word in comment:
            word = clean_str(word)
            if word_index >= sentence_length:
                break
            try:
                word_embedding = word_to_vector[word]
            except:
                continue
            sample[word_index] = word_embedding.reshape((word_dimension, 1))
            word_index += 1
        samples[sample_index] = sample
    return samples


def create_1vec_labels(labels, labels_values):
    amount_samples = labels_values.shape[0]
    onevec_labels = np.zeros((amount_samples, len(labels)), dtype=np.float32)
    for index in range(amount_samples):
        onevec_labels[index, labels.index(labels_values[index])] = 1
    return onevec_labels


def load_comments(comments_path):
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

    comments_rating_1 = comments_frame[comments_frame['rating'] == 1][0:50]
    comments_rating_2 = comments_frame[comments_frame['rating'] == 2][0:50]
    comments_rating_3 = comments_frame[comments_frame['rating'] == 3][0:50]
    comments_rating_4 = comments_frame[comments_frame['rating'] == 4][0:50]
    comments_rating_5 = comments_frame[comments_frame['rating'] == 5][0:50]

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
    EMBEDDING_PATH = 'raw/SBW-vectors-300-min5.bin'
    COMMENTS_PATH = 'raw/comments.csv'
    sentence_length = 100
    channels = 1
    raw_labels = [1, 2, 3, 4, 5]

    print('loading embedding')
    embedding = load_embedding(EMBEDDING_PATH)
    print('loading comments')
    comments_frame = load_comments(COMMENTS_PATH)

    samples_count, _ = comments_frame.shape
    _, word_dimension = embedding.wv.vectors.shape

    np.random.seed(0)
    comments_frame = comments_frame.iloc[np.random.permutation(samples_count)].reset_index(drop=True)

    print('processing comments')
    samples = process_comments(
        embedding.wv,
        comments_frame['fullContent'],
        samples_count,
        sentence_length,
        word_dimension,
        channels)
    onevec_labels = create_1vec_labels(raw_labels, comments_frame['rating'])

    # Preparation
    assert samples.shape[0] == onevec_labels.shape[0]
    seventyfive_percent = math.floor(samples.shape[0] * 0.75)
    fifteen_percent = math.floor(samples.shape[0] * 0.15)
    train_dataset = samples[:seventyfive_percent]
    train_labels = onevec_labels[:seventyfive_percent]
    val_dataset = samples[seventyfive_percent:seventyfive_percent + fifteen_percent]
    val_labels = onevec_labels[seventyfive_percent:seventyfive_percent + fifteen_percent]
    test_dataset = samples[seventyfive_percent + fifteen_percent:]
    test_labels = onevec_labels[seventyfive_percent + fifteen_percent:]
    print('train dataset shape: ({}, {}, {}, {})'.format(*train_dataset.shape))
    print('train labels shape: ({}, {})'.format(*train_labels.shape))
    print('val dataset shape: ({}, {}, {}, {})'.format(*val_dataset.shape))
    print('val labels shape: ({}, {})'.format(*val_labels.shape))
    print('test dataset shape: ({}, {}, {}, {})'.format(*test_dataset.shape))
    print('test labels shape: ({}, {})'.format(*test_labels.shape))
    ## TODO: display the next with graphics
    print('Data ratings : ' + str(raw_labels))
    print('------ train : ' + str(np.sum(train_labels, axis=0)))
    print('------ val   : ' + str(np.sum(val_labels, axis=0)))
    print('------ test  : ' + str(np.sum(test_labels, axis=0)))

    # Save
    DataManager.save_data(
        train_dataset,
        train_labels,
        val_dataset,
        val_labels,
        test_dataset,
        test_labels)
    print('train and test data saved at data/')


if __name__ == '__main__':
    _main()
