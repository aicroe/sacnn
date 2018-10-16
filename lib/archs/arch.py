from abc import ABC, abstractmethod
from lib.data_saver import DataSaver
import tensorflow as tf


class Arch(ABC):

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def initialize_variables(self, session):
        pass

    @abstractmethod
    def _get_variables_as_list(self):
        pass

    def save(self, session, name):
        arch_name = type(self).__name__.lower()
        full_path = str(DataSaver.prepare_dir(arch_name).joinpath('%s.ckpt' % name))
        saver = tf.train.Saver(self._get_variables_as_list())
        saver.save(session, full_path)
        return full_path

    def restore(self, session, name):
        arch_name = type(self).__name__.lower()
        full_path = str(DataSaver.prepare_dir(arch_name).joinpath('%s.ckpt' % name))
        restorer = tf.train.Saver(self._get_variables_as_list())
        restorer.restore(session, full_path)
        return full_path
