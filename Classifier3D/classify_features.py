import sys
import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Classifier3D.utils.ply_file import read_ply
from Classifier3D.preprocessing.n_per_class import get_n_points_per_class
from Classifier3D.feature_extraction.extractor import compute_features
from Classifier3D.utils.submission import save_prediction

from Classifier3D.feature_extraction import ALL_FEATURES


def classify_features_cli(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Pipeline to classify the 3D points by computing features")
    parser.add_argument(
        "-trainf",
        "--train_files",
        default=["MiniLille1.ply", "MiniLille2.ply", "MiniParis1.ply"],
        help="List of train files, (default: ['MiniLille1.ply', 'MiniLille2.ply', 'MiniParis1.ply']).",
    )
    parser.add_argument(
        "-testf",
        "--test_file",
        type=str,
        default="MiniDijon9.ply",
        help="Test file, (default: 'MiniDijon9.ply').",
    )
    parser.add_argument(
        "-v",
        "--verticality",
        default=False,
        action="store_true",
        help="if given, verticality will be computed, (default: False).",
    )
    parser.add_argument(
        "-l",
        "--linearity",
        default=False,
        action="store_true",
        help="if given, linearity will be computed, (default: False).",
    )
    parser.add_argument(
        "-p",
        "--planarity",
        default=False,
        action="store_true",
        help="if given, planarity will be computed, (default: False).",
    )
    parser.add_argument(
        "-s",
        "--sphericity",
        default=False,
        action="store_true",
        help="if given, sphericity will be computed, (default: False).",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=None,
        help="The radius to define neighborhood. If None, k_neighbors will be considered, (default: None).",
    )
    parser.add_argument(
        "-k",
        "--k_neighbors",
        type=int,
        default=None,
        help="The number of neighbors to define neighborhood. If None, radius will be considered, (default: None).",
    )
    parser.add_argument(
        "-npc",
        "--number_per_class",
        type=int,
        default=500,
        help="The number of points that will be taken per class (default: 500).",
    )
    parser.add_argument(
        "-ns",
        "--name_submission",
        required=True,
        help="The nume of the submission file. '.txt' with be added at the end.",
    )
    args = parser.parse_args(argvs)

    args.data_files = {}
    for train_file in args.train_files:
        args.data_files[train_file] = True
    args.data_files[args.test_file] = False

    args.features = {}
    args.n_features = 0
    for feature in ALL_FEATURES:
        args.features[feature] = args[feature]
        args.n_features += int(args[feature])

    assert (
        args.radius is not None and args.k_neighbors is not None
    ), "You should give either radius or k_neibors but not both"
    if args.radius is not None:
        args.radius_or_k_neighbors = args.radius
        args.use_radius = True
    elif args.k_neighbors is not None:
        args.radius_or_k_neighbors = args.k_neighbors
        args.use_radius = False
    else:
        raise Error("You should give either radius or k_neibors but at least one")

    print(args)

    classify(args)


def classify(args):
    train_features = np.empty((0, args.n_features))
    train_labels = np.empty((0))
    test_features = np.empty((0, args.n_features))

    for file_path, is_train_data in args.data_files.items():
        # Loading file
        cloud = read_ply(file_path)
        points = np.vstack((cloud["x"], cloud["y"], cloud["z"])).T
        if is_train_data:
            labels = cloud["class"]

        # Preproccess
        if is_train_data:
            chosen_indexes = get_n_points_per_class(labels, args.number_per_class)
        else:
            chosen_indexes = np.arange(points.shape[0])

        # Feature extraction
        features = compute_features(
            points[chosen_indexes], points, args.radius_or_k_neighbors, args.use_radius, **args.features
        )

        # Update arrays
        if is_train_data:
            train_features = np.vstack((train_features, features))
            train_labels = np.vstack((train_labels, labels[chosen_indexes]))
        else:
            test_features = np.vstack((test_features, features))

    # Classify
    classifier = RandomForestClassifier()

    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    # Regularize
    # Not implemented

    # Save submission file
    save_prediction(args.name_submission, predictions)
