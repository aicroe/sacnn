import numpy as np
from lib.singleton_decorator import singleton


@singleton()
class Accuracy(object):
    def __call__(_, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / predictions.shape[0])
