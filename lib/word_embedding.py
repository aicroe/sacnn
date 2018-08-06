from gensim.models import KeyedVectors
from .data_saver import DataSaver


class WordEmbedding(object):
    embedding = None

    @staticmethod
    def get_instance(path=str(DataSaver.prepare_dir('raw').joinpath('SBW-vectors-300-min5.bin'))):
        # other word2vecs binaries are loaded as:
        # from gensim.models import Word2Vec
        # Word2Vec.load(...)
        if WordEmbedding.embedding is None:
            WordEmbedding.embedding = KeyedVectors.load_word2vec_format(path, binary=True)
        _, word_dimension = WordEmbedding.embedding.wv.vectors.shape
        return WordEmbedding.embedding, word_dimension
