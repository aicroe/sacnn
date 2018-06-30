from gensim.models import KeyedVectors
from .data_saver import DataSaver


class EmbeddingFactory(object):
    embedding = None

    @staticmethod
    def get_embedding(path=str(DataSaver.prepare_dir('raw').joinpath('SBW-vectors-300-min5.bin'))):
        # other word2vecs binaries are loaded as:
        # from gensim.models import Word2Vec
        # Word2Vec.load(...)
        if EmbeddingFactory.embedding is None:
            embedding = KeyedVectors.load_word2vec_format(path, binary=True)
        _, word_dimension = embedding.wv.vectors.shape
        return embedding, word_dimension
