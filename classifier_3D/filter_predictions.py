import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def filter_predictions_cli(argvs=sys.argv[1:]):
    import argparse

    from classifier_3D.utils.path import get_data_path, get_submission_path

    parser = argparse.ArgumentParser(
        "Pipeline to filter the predictions of the 3D points."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="File on which to compute the features, (required).",
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
        help="Save the file with the filtered classification labels. (default: False)",
    )
    args = parser.parse_args(argvs)
    args = vars(args)

    args["file_path"] = get_data_path(args["file"], is_train_data=False)

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

    args["path_submission"] = get_submission_path(args["name_submission"])

    print(args)

    filter(args)


def filter(args):
    from classifier_3D.utils.ply_file import read_ply, write_ply
    from classifier_3D.utils.neighbors import compute_index_neighbors
    from classifier_3D.utils.submission import save_prediction
    from classifier_3D.metric.confusion_matrix import get_confusion_matrix
    from classifier_3D.metric.IoU import get_IoU

    from classifier_3D import LABEL_NAMES

    # Loading file
    cloud, headers = read_ply(args["file_path"])

    points = np.vstack((cloud["x"], cloud["y"], cloud["z"])).T.astype(np.float32)
    labels = cloud["prediction"]

    # Filter
    # Remove the points that are classified as ground and that have a height lower that zero
    index_no_ground = ~np.logical_and(labels == 1, cloud["z"] <= 0)
    points_no_ground = points[index_no_ground]
    label_no_ground = labels[index_no_ground]

    print(f"Compute the neighbors of {len(points_no_ground)} points.")
    idx_neighbors_queries = compute_index_neighbors(
        points_no_ground,
        points_no_ground,
        args["radius_or_k_neighbors"],
        args["use_radius"],
    )

    new_labels = []
    for idx_neighbors in tqdm(idx_neighbors_queries):
        new_labels.append(
            np.floor(np.median(label_no_ground[idx_neighbors])).astype(int)
        )

    predictions = labels.copy()
    predictions[index_no_ground] = new_labels

    # Save submission file
    save_prediction(args["path_submission"], predictions)

    if args["save_classification"]:
        structured_cloud = np.vstack(
            [cloud[header] for header in headers if header != "prediction"]
        ).T
        headers.remove("prediction")

        write_ply(
            args["file_path"].replace(".ply", f"_{args['name_submission']}.ply"),
            [structured_cloud, predictions.astype(np.int32)],
            headers + ["prediction"],
        )

    # Check if we can compute the confusion matrix
    if "class" in headers:
        old_confusion_matrix = get_confusion_matrix(cloud["prediction"], cloud["class"])

        print("\n\n\n")
        print(f"The old confusion matrix for {args['file_path']} is:")
        print(
            pd.DataFrame(
                data=old_confusion_matrix,
                columns=list(LABEL_NAMES.values())[1:],
                index=list(LABEL_NAMES.values())[1:],
            )
        )

        IoUs = []
        for label in list(LABEL_NAMES.keys())[1:]:
            IoUs.append(get_IoU(old_confusion_matrix, label))

        print("\nThe new IoUs are:")
        print(pd.Series(data=IoUs, index=list(LABEL_NAMES.values())[1:]))
        print(f"New average IoU: {np.around(np.mean(IoUs), 3)}")

        new_confusion_matrix = get_confusion_matrix(predictions, cloud["class"])

        print("\n\n")
        print(f"The new confusion matrix for {args['file_path']} is:")
        print(
            pd.DataFrame(
                data=new_confusion_matrix,
                columns=list(LABEL_NAMES.values())[1:],
                index=list(LABEL_NAMES.values())[1:],
            )
        )

        IoUs = []
        for label in list(LABEL_NAMES.keys())[1:]:
            IoUs.append(get_IoU(new_confusion_matrix, label))

        print("\nThe new IoUs are:")
        print(pd.Series(data=IoUs, index=list(LABEL_NAMES.values())[1:]))
        print(f"New average IoU: {np.around(np.mean(IoUs), 3)}")
