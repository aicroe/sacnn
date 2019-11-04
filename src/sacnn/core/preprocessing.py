import re
import numpy as np

CHANNELS = 1

def clean_word(dirty_word):
    return re.sub(r'[\.,\-"\{\}\[\]\*\^;%\+&°!¡¿?#<>@/\(\)\\=:_~\$€\|]', '', dirty_word.lower())

def create_1hot_vectors(values, labels):
    batch_size, *_ = values.shape
    onehot_vectors = np.zeros((batch_size, len(labels)), dtype=np.float32)
    for index in range(batch_size):
        onehot_vectors[index, labels.index(values[index])] = 1

    return onehot_vectors

def process_sentence(text, sentence_length, word_dimension, word_to_vector):
    sentence = np.zeros((sentence_length, word_dimension, CHANNELS))
    for index, word in enumerate(text.split()):
        cleaned_word = clean_word(word)
        if index >= sentence_length:
            break
        try:
            word_embedding = word_to_vector[cleaned_word]
        except:
            continue
        sentence[index] = word_embedding.reshape(word_dimension, CHANNELS)

    return sentence

def process_sentences(texts, sentence_length, word_dimension, word_to_vector):
    texts_array = np.array(texts)
    texts_number, *_ = texts_array.shape
    sentences = np.empty(
        (texts_number, sentence_length, word_dimension, CHANNELS),
        dtype=np.float32,
    )
    for index in range(texts_number):
        sentences[index] = process_sentence(
            texts_array[index],
            sentence_length,
            word_dimension,
            word_to_vector,
        )

    return sentences
