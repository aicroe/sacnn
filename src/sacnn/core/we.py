from gensim.models import KeyedVectors
#from gensim.models import Word2Vec

from .fs_utils import prepare_dir


def get_word_to_vector(path=prepare_dir('raw').joinpath('SBW-vectors-300-min5.bin'), limit=1000):
#def _get_word_to_vector(path=str(prepare_dir('raw-alt').joinpath('es.bin'))):
    #embedding = Word2Vec.load(path)
    embedding = KeyedVectors.load_word2vec_format(path, binary=True, limit=limit)
    #_, word_dimension = embedding.wv.syn0.shape
    _, word_dimension = embedding.syn0.shape
    return embedding, word_dimension
