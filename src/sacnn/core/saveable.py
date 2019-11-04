from abc import ABC, abstractmethod
import tensorflow as tf

from .fs_utils import prepare_dir

def save_tf_instance(name, dir_name, variables, session):
    save_dir = prepare_dir(dir_name)
    full_path = str(save_dir.joinpath('%s.ckpt' % name))
    saver = tf.compat.v1.train.Saver(variables)
    saver.save(session, full_path)
    return full_path

def restore_tf_instance(name, dir_name, variables, session):
    save_dir = prepare_dir(dir_name)
    full_path = str(save_dir.joinpath('%s.ckpt' % name))
    restorer = tf.compat.v1.train.Saver(variables)
    restorer.restore(session, full_path)
    return full_path

class Saveable(ABC):

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def restore(self):
        pass
