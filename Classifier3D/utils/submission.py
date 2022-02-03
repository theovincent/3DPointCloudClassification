import numpy as np

from Classifier3D import NUMBER_TEST_POINTS


def save_prediction(name, predictions):
    assert type(name) == str
    assert np.ndim(predictions) == 1
    assert predictions.shape[0] == NUMBER_TEST_POINTS

    np.savetxt(f"submissions/{name}.txt", predictions, fmt="%d")
