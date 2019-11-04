import numpy as np
from mlscratch import Model

def get_sentiment_map(num_labels):
    if num_labels == 3:
        return {
            1: 'Negative',
            2: 'Neutral',
            3: 'Positive'
        }
    elif num_labels == 5:
        return {
            1: 'Terrible',
            2: 'Bad',
            3: 'Neutral',
            4: 'Good',
            5: 'Excellent'
        }

class SacnnModel(Model):

    def __init__(self, arch):
        super().__init__(arch)
        self._arch = arch

    def get_arch_name(self):
        return self._arch.get_name()

    def save(self):
        self._arch.save()

    def restore(self):
        self._arch.restore()

    def sentiment(self, dataset):
        evaluation = self.evaluate(dataset)
        num_samples, num_labels = evaluation.shape

        max_indices = np.argmax(evaluation, axis=1)
        predicted_labels = max_indices + 1
        labels_probs = evaluation[range(num_samples), max_indices]
        sentiment_map = get_sentiment_map(num_labels)

        return [
            {
                'sentiment': sentiment_map[predicted_label],
                'prob': str(label_prob * 100),
            }
            for (predicted_label, label_prob)
            in zip(predicted_labels, labels_probs)
        ]
