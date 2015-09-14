class ConfusionMatrix:
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def true_negatives(self):
        return self.matrix[0][0]

    @property
    def false_positives(self):
        return self.matrix[0][1]

    @property
    def true_positives(self):
        return self.matrix[1][1]

    @property
    def false_negatives(self):
        return self.matrix[1][0]


def recall(confusion_matrix):
    return confusion_matrix.true_positives/(confusion_matrix.true_positives + confusion_matrix.false_negatives)


def false_positive_rate(confusion_matrix):
    return confusion_matrix.false_positives/(confusion_matrix.false_positives + confusion_matrix.true_negatives)
