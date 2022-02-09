import numpy as np


def save_prediction(path, predictions):
    assert np.ndim(predictions) == 1

    np.savetxt(path, predictions, fmt="%d")
