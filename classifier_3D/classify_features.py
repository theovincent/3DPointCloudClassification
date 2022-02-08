import sys

import numpy as np


def classify_features_cli(argvs=sys.argv[1:]):
    import argparse

    from classifier_3D.utils.path import get_data_path, get_submission_path

    from classifier_3D.feature_extraction import ALL_FEATURES

    parser = argparse.ArgumentParser("Pipeline to classify the 3D points by computing features")
    parser.add_argument(
        "-trainf",
        "--train_files",
        nargs="+",
        default=["MiniLille1_with_features.ply", "MiniLille2_with_features.ply", "MiniParis1_with_features.ply"],
        help="List of train files, (default: ['MiniLille1_with_features.ply', 'MiniLille2_with_features.ply', 'MiniParis1_with_features.ply']).",
    )
    parser.add_argument(
        "-testf",
        "--test_file",
        type=str,
        default="MiniDijon9_with_features.ply",
        help="Test file, (default: 'MiniDijon9_with_features.ply').",
    )
    parser.add_argument(
        "-v",
        "--verticality",
        default=False,
        action="store_true",
        help="if given, verticality will be used for classification, (default: False).",
    )
    parser.add_argument(
        "-l",
        "--linearity",
        default=False,
        action="store_true",
        help="if given, linearity will be used for classification, (default: False).",
    )
    parser.add_argument(
        "-p",
        "--planarity",
        default=False,
        action="store_true",
        help="if given, planarity will be used for classification, (default: False).",
    )
    parser.add_argument(
        "-s",
        "--sphericity",
        default=False,
        action="store_true",
        help="if given, sphericity will be used for classification, (default: False).",
    )
    parser.add_argument(
        "-x",
        "--x_coordinate",
        default=False,
        action="store_true",
        help="if given, the x coordinate will be used for classification, (default: False).",
    )
    parser.add_argument(
        "-y",
        "--y_coordinate",
        default=False,
        action="store_true",
        help="if given, the y coordinate will be used for classification, (default: False).",
    )
    parser.add_argument(
        "-z",
        "--z_coordinate",
        default=False,
        action="store_true",
        help="if given, the z coordinate will be used for classification, (default: False).",
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
        help="The nume of the submission file. (required)",
    )
    parser.add_argument(
        "-sc",
        "--save_classification",
        default=False,
        action="store_true",
        help="Save the test file with the classification labels. (default: False)",
    )
    args = parser.parse_args(argvs)
    args = vars(args)

    args["data_files"] = {}
    for train_file in args["train_files"]:
        train_file_path = get_data_path(train_file, is_train_data=True)
        args["data_files"][train_file_path] = True

    test_file_path = get_data_path(args["test_file"], is_train_data=False)
    args["data_files"][test_file_path] = False

    args["features"] = []
    for feature in ALL_FEATURES:
        if args[feature]:
            args["features"].append(feature)

    args["path_submission"] = get_submission_path(args["name_submission"])

    print(args)

    classify(args)


def classify(args):
    from sklearn.ensemble import RandomForestClassifier

    from classifier_3D.utils.ply_file import read_ply, write_ply
    from classifier_3D.preprocessing.n_per_class import get_n_points_per_class
    from classifier_3D.utils.submission import save_prediction

    train_features_list = []
    train_labels_list = []
    test_features = None

    print("Load file and get the features")
    for file_path, is_train_data in args["data_files"].items():
        # Loading file
        cloud, _ = read_ply(file_path)
        features = np.vstack([cloud[feature] for feature in args["features"]]).T
        if is_train_data:
            labels = cloud["class"]

        # Preproccess
        if is_train_data:
            chosen_indexes = get_n_points_per_class(labels, args["number_per_class"])
        else:
            chosen_indexes = np.arange(features.shape[0])

        # Update arrays
        if is_train_data:
            train_features_list.append(features[chosen_indexes])
            train_labels_list.append(labels[chosen_indexes])
        else:
            test_features = features[chosen_indexes]

    train_features = np.vstack(train_features_list)
    train_labels = np.hstack(train_labels_list)

    # Classify
    print(f"Classify features, training set composed of {len(train_labels)} points.")
    classifier = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        min_samples_split=50,
        min_samples_leaf=20,
        bootstrap=True,
        n_jobs=4,
        class_weight="balanced_subsample",
        max_samples=0.4,
    )

    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    # Regularize
    # Not implemented

    # Save submission file
    save_prediction(args["path_submission"], predictions)

    if args["save_classification"]:
        for file_path, is_train_data in args["data_files"].items():
            if is_train_data:
                continue

            cloud, headers = read_ply(file_path)

            structured_cloud = np.vstack([cloud[header] for header in headers]).T

            write_ply(
                file_path.replace(".ply", f"_{args['name_submission']}.ply"),
                [structured_cloud, predictions.astype(np.int32)],
                headers + ["prediction"],
            )
