from sacnn.core import SacnnModel, get_arch
from sacnn.core.preprocessing import process_sentences

from .instance_stack import InstanceStack
from .app_state import app_state
from .constraints import FILTERS, SENTENCE_LENGTH
from .word_to_vector import word_to_vector, WORD_DIMENSION


class ClassifierController():

    def __init__(self, max_stack=2):
        """
        :param int max_stack:
        """
        self._instance_stack = InstanceStack(max_stack)

    def select_instance(self, instance_name):
        """
        :param str instance_name:
        """
        if instance_name not in self._instance_stack:
            instance_data = app_state.get_instance_by_name(instance_name)

            if instance_data is not None:
                (name, hidden_units, num_labels, arch) = instance_data

                hyperparams = {
                    'name': instance_name,
                    'arch': arch,
                    'sentence_length': SENTENCE_LENGTH,
                    'word_dimension': WORD_DIMENSION,
                    'hidden_units': int(hidden_units),
                    'filters': FILTERS,
                    'num_labels': int(num_labels),
                }
                model = SacnnModel(get_arch(hyperparams))
                model.restore()
                self._instance_stack.push(model)
            else:
                raise BaseException('instance_not_found_by_name')

    def classify(self, instance_name, comments):
        """
        :param str instance_name:
        :param str comment:
        :returns str:
        """
        self.select_instance(instance_name)
        instance = self._instance_stack[instance_name]
        input_data = process_sentences(comments, SENTENCE_LENGTH, WORD_DIMENSION, word_to_vector)

        return instance.sentiment(input_data)
