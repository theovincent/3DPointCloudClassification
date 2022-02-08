def get_IoU(confusion_matrix, label):
    # There are -1 everywhere since "Unclassified" is not counted.
    return confusion_matrix[label - 1, label - 1] / (
        confusion_matrix[label - 1, :].sum() + confusion_matrix[:, label - 1].sum() - confusion_matrix[label - 1, label - 1]
    )
