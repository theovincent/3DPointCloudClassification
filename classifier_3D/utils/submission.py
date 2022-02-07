import numpy as np

from classifier_3D import NUMBER_TEST_POINTS


def save_prediction(path, predictions):
    assert np.ndim(predictions) == 1
    # assert predictions.shape[0] == NUMBER_TEST_POINTS

    np.savetxt(path, predictions, fmt="%d")
