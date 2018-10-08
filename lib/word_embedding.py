from gensim.models import KeyedVectors
#from gensim.models import Word2Vec
from .data_saver import DataSaver


class WordEmbedding(object):
    embedding = None

    @staticmethod
    def get_instance(path=str(DataSaver.prepare_dir('raw').joinpath('SBW-vectors-300-min5.bin')), limit=100000):
    #def get_instance(path=str(DataSaver.prepare_dir('raw-alt').joinpath('es.bin'))):
        if WordEmbedding.embedding is None:
            #WordEmbedding.embedding = Word2Vec.load(path)
            WordEmbedding.embedding = KeyedVectors.load_word2vec_format(path, binary=True, limit=limit)
        #_, word_dimension = WordEmbedding.embedding.wv.syn0.shape
        _, word_dimension = WordEmbedding.embedding.syn0.shape
        return WordEmbedding.embedding, word_dimension
