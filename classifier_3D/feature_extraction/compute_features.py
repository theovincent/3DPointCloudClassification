import sys

import numpy as np


def compute_features_cli(argvs=sys.argv[1:]):
    import argparse

    from classifier_3D.utils.path import get_data_path

    from classifier_3D.feature_extraction import ALL_FEATURES

    parser = argparse.ArgumentParser("Pipeline to compute features on 3D points")
    parser.add_argument("-f", "--file", required=True, help="File on which to compute the features, (required).")
    parser.add_argument(
        "-itd",
        "--is_train_data",
        default=False,
        action="store_true",
        help="Nature of the file on which to compute the features, (default=False).",
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
        help="The radius to define neighborhood in meters. If None, k_neighbors will be considered, (default: None).",
    )
    parser.add_argument(
        "-k",
        "--k_neighbors",
        type=int,
        default=None,
        help="The number of neighbors to define neighborhood. If None, radius will be considered, (default: None).",
    )
    args = parser.parse_args(argvs)
    args = vars(args)

    args["file_path"] = get_data_path(args["file"], is_train_data=args["is_train_data"])

    args["features"] = {}
    for feature in ALL_FEATURES:
        if feature in ["x", "y", "z"]:
            continue
        args["features"][feature] = args[feature]

    assert not (
        args["radius"] is not None and args["k_neighbors"] is not None
    ), "You should give either radius or k_neibors but not both"
    if args["radius"] is not None:
        args["radius_or_k_neighbors"] = args["radius"]
        args["use_radius"] = True
    elif args["k_neighbors"] is not None:
        args["radius_or_k_neighbors"] = args["k_neighbors"]
        args["use_radius"] = False
    else:
        raise ValueError("You should give either radius or k_neibors but at least one")

    print(args)

    compute(args)


def compute(args):
    from classifier_3D.utils.ply_file import read_ply, write_ply
    from classifier_3D.feature_extraction.extractor import get_features

    # Loading file
    cloud, headers = read_ply(args["file_path"])

    points = np.vstack((cloud["x"], cloud["y"], cloud["z"])).T

    # Feature extraction
    features, features_headers = get_features(
        points, points, args["radius_or_k_neighbors"], args["use_radius"], **args["features"]
    )

    structured_cloud = np.vstack([cloud[header] for header in headers if header not in features_headers]).T
    pruned_headers = [header for header in headers if header not in features_headers]

    write_ply(
        args["file_path"].replace(".ply", "_with_features.ply"),
        [structured_cloud, features],
        pruned_headers + features_headers,
    )
