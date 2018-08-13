from abc import ABC
from lib.base_sacnn_builder import BaseSCANNBuilder
from lib.evolved_sacnn_builder import EvolvedCANNBuilder
from lib.word_embedding import WordEmbedding


class AppController(ABC):
    sentence_length = 100
    filters_size = [(3, 96), (5, 96), (7, 64)]
    (_, word_dimension) = WordEmbedding.get_instance()
    builder = {
        'base': BaseSCANNBuilder.get_instance(),
        'evolved': EvolvedCANNBuilder.get_instance()
    }
