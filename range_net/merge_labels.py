import os
import sys
import numpy as np
from tqdm import tqdm


def merge_labels_cli(argvs=sys.argv[1:]):
    import argparse

    from classifier_3D.utils.ply_file import read_ply, write_ply
    from classifier_3D.utils.path import get_data_path

    from range_net import (
        CITY_INFERANCE_FOLDER,
        PATH_INDEXES_TO_KEEP,
        KITTI_TO_CITY_NUMBERS,
        CITY_NUMBERS_TO_LABELS,
        MAX_DISTANCE,
    )

    parser = argparse.ArgumentParser("Pipeline to merge the samples from RangeNet++.")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="File on which to sample points, (required).",
    )
    parser.add_argument(
        "-itd",
        "--is_train_data",
        default=False,
        action="store_true",
        help="Tell the algorithm if the file is a training data. (default: False)",
    )
    parser.add_argument(
        "-pp",
        "--path_predictions",
        type=str,
        default="/home/theovincent/MVA/3DPointCloud/RangeNet++/lidar-bonnetal/train/tasks/semantic/preds/sequences/FOLDER/predictions",
        help="Path where to store the .bins and the .labels file, (default: /home/theovincent/MVA/3DPointCloud/RangeNet++/lidar-bonnetal/train/tasks/semantic/preds/sequences/FOLDER/predictions",
    )
    args = parser.parse_args(argvs)
    args = vars(args)
    print(args)

    path_to_predictions = args["path_predictions"].replace(
        "FOLDER", CITY_INFERANCE_FOLDER[args["file"]]
    )

    point_cloud_path = get_data_path(
        f"{args['file']}_with_features", args["is_train_data"]
    )
    point_cloud, _ = read_ply(point_cloud_path)
    points = np.vstack((point_cloud["x"], point_cloud["y"], point_cloud["z"])).T.astype(
        np.float32
    )

    list_path_predictions = sorted(
        [
            f"{path_to_predictions}/{file_name}"
            for file_name in os.listdir(path_to_predictions)
            if file_name[-6:] == ".label"
        ]
    )

    weighted_targets = np.zeros(
        (points.shape[0], len(CITY_NUMBERS_TO_LABELS)), dtype=np.float32
    )

    for path_prediction in tqdm(list_path_predictions):
        path_to_indexes = f"{PATH_INDEXES_TO_KEEP}/{CITY_INFERANCE_FOLDER[args['file']]}/{os.path.split(path_prediction)[-1].replace('.label', '.npy')}"
        path_to_sample = (
            path_prediction.replace("preds", "data_city")
            .replace("predictions", "velodyne")
            .replace(".label", ".bin")
        )

        indexes = np.load(path_to_indexes).astype(bool)
        sample = np.fromfile(path_to_sample, dtype=np.float32).reshape((-1, 4))[:, :3]
        ranges = np.linalg.norm(sample, axis=1).astype(np.float32)
        kitti_prediciton = np.fromfile(path_prediction, dtype=np.int32)
        city_prediciton = np.array(
            [KITTI_TO_CITY_NUMBERS[label] for label in kitti_prediciton]
        ).astype(np.int32)

        weights = np.exp(-10 * ranges / MAX_DISTANCE)

        weighted_targets[indexes, city_prediciton] += weights

    write_ply(
        point_cloud_path.replace(
            f"{args['file']}_with_features",
            f"{args['file']}_with_range_net_{len(list_path_predictions)}_samples",
        ),
        (points, weighted_targets.argmax(axis=1).astype(np.int32)),
        ["x", "y", "z", "class"],
    )
