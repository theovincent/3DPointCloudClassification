import numpy as np

from classifier_3D import LABEL_NAMES


def get_n_points_per_class(labels, number_per_class):
    indexes = []

    for label in LABEL_NAMES:
        # Do not include class 0 in training
        if label == 0:
            continue

        # Collect all indices of the current class
        label_indexes = np.where(labels == label)[0]

        # If you have not enough indices, just take all of them
        if len(label_indexes) <= number_per_class:
            indexes.append(label_indexes)
        else:
            random_choice = np.random.choice(
                len(label_indexes), number_per_class, replace=False
            )
            indexes.append(label_indexes[random_choice])

    # Gather chosen points
    return np.hstack(indexes)
