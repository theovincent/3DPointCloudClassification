def get_IoU(confusion_matrix, label):
    assert label > 0 and label < 7, f"The given label {label} has to be between 1 and 6."

    if confusion_matrix[label - 1, :].sum() == 0:
        return None

    # There are -1 everywhere since "Unclassified" is not counted.
    return confusion_matrix[label - 1, label - 1] / (
        confusion_matrix[label - 1, :].sum()
        + confusion_matrix[:, label - 1].sum()
        - confusion_matrix[label - 1, label - 1]
    )
