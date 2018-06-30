import numpy as np


class CommentClassifier(object):
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor

    def classify(self, comments):
        data = self.data_processor.process(comments)
        return self.model.evaluate(data)

    def classify_one(self, comment):
        return self.classify(np.array([comment]))
