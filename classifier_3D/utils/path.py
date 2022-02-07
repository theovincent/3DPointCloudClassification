import os

from classifier_3D import DATA_PATH, SUBMISSION_PATH


def get_data_path(name, is_train_data):
    if name[-4:] != ".ply":
        name += ".ply"

    if is_train_data:
        path = f"{DATA_PATH}/train/{name}"
    else:
        path = f"{DATA_PATH}/test/{name}"

    assert os.path.exists(path), f"The path {path} does not exist."

    return path


def get_submission_path(name):
    if name[-4:] != ".txt":
        name += ".txt"

    path = f"{SUBMISSION_PATH}/{name}"

    assert not os.path.exists(path), f"The path {path} for submission already exists."

    return path
