import numpy as np
from tqdm import tqdm

from classifier_3D import LABEL_NAMES


def get_confusion_matrix(predictions, labels):
    # There are -1 everywhere since "Unclassified" is not counted.
    confusion = np.zeros((len(LABEL_NAMES.keys()) - 1, len(LABEL_NAMES.keys()) - 1))

    for idx_predicted_label, predicted_label in tqdm(enumerate(LABEL_NAMES.keys()[1:])):
        predicted_label_index = predictions == predicted_label

        for idx_true_label, true_label in enumerate(LABEL_NAMES.keys()[1:]):
            true_label_index = labels == true_label

            confusion[idx_true_label, idx_predicted_label] = (predicted_label_index & true_label_index).sum()

    return confusion
