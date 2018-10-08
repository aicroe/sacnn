import numpy as np
import re
import math
from gensim.models import KeyedVectors
import numpy as np


class DataProcessor(object):

    @staticmethod
    def create_1hot_vectors(values, possible_values):
        (num_values,) = values.shape
        onehot_values = np.zeros(
            (num_values, len(possible_values)), dtype=np.float32)
        for index in range(num_values):
            onehot_values[index, possible_values.index(values[index])] = 1
        return onehot_values

    @staticmethod
    def clean_word(dirty_word):
        return re.sub(r'[\.,\-"\{\}\[\]\*\^;%\+&°!¡¿?#<>@/\(\)\\=:_~\$€\|]', '', dirty_word.lower())

    def __init__(self,
                 word_to_vector,
                 sentence_length,
                 word_dimension,
                 channels):
        self.word_to_vector = word_to_vector
        self.sentence_length = sentence_length
        self.word_dimension = word_dimension
        self.channels = channels

    def process_one(self, sentence):
        sample = np.zeros(
            (self.sentence_length, self.word_dimension, self.channels))
        word_index = 0
        for word in sentence.split():
            word = self.clean_word(word)
            if word_index >= self.sentence_length:
                break
            try:
                word_embedding = self.word_to_vector[word]
            except:
                continue
            sample[word_index] = word_embedding.reshape(
                (self.word_dimension, self.channels))
            word_index += 1
        return sample

    def process(self, sentences):
        (sentences_count,) = sentences.shape
        samples = np.empty((sentences_count, self.sentence_length,
                            self.word_dimension, self.channels), dtype=np.float32)
        for sample_index in range(sentences_count):
            samples[sample_index] = self.process_one(sentences[sample_index])
        return samples
