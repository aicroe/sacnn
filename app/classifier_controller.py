from lib.sacnn import SACNN
from lib.word_embedding import WordEmbedding
from lib.data_processor import DataProcessor
from lib.sacnn_builders.sacnn_creator import SACNNCreator
from .instance_stack import InstanceStack
from .app_state import AppState
from .app_controller import AppController
import types


def sentiment_factory(sentiment_map):
    """
    :param dict sentiment_map
    """ 
    def evaluate(self, input_data):
        """
        :param SACNN self:
        :param numpy input_data:
        """
        return list(
            map(lambda evaluation: {
                'sentiment': sentiment_map[evaluation[0]],
                'probability': str(evaluation[1] * 100),
            }, self.evaluate(input_data))
        )
    return evaluate


class ClassifierController(AppController):
    sentiment = {
        3: sentiment_factory({
            1: 'Negativo',
            2: 'Regular',
            3: 'Positivo'
        }),
        5: sentiment_factory({
            1: 'Terrible',
            2: 'Malo',
            3: 'Regular',
            4: 'Bueno',
            5: 'Excelente'
        })
    }

    def __init__(self, app_state, max_stack=2):
        """
        :param AppState app_state:
        :param int max_stack:
        """
        self.instance_stack = InstanceStack(max_stack)
        self.word_embedding, word_dimension = WordEmbedding.get_instance()
        self.data_processor = DataProcessor(self.word_embedding,
                                            self.sentence_length,
                                            word_dimension,
                                            SACNNCreator.channels)
        self.app_state = app_state

    def select_instance(self, instance_name):
        """
        :param str instance_name:
        """
        if instance_name not in self.instance_stack:
            instance_data = self.app_state.get_instance_by_name(instance_name)
            if instance_data is not None:
                (_, hidden_units, num_labels, arch) = instance_data
                hyperparams = {
                    'name': instance_name,
                    'sentence_length': self.sentence_length,
                    'word_dimension': self.word_dimension,
                    'hidden_units': int(hidden_units),
                    'filters_size': self.filters_size,
                    'num_labels': int(num_labels)
                }
                instance = self.builders[arch].restore(hyperparams)
                instance.sentiment = types.MethodType(self.sentiment[int(num_labels)], instance)
                self.instance_stack.push(instance)
            else:
                raise BaseException('instance_not_found_by_name')

    def classify(self, instance_name, comments):
        """
        :param str instance_name:
        :param str comment:
        :returns str:
        """
        self.select_instance(instance_name)
        instance = self.instance_stack[instance_name]
        input_data = self.data_processor.process(comments)
        return instance.sentiment(input_data)
