import os
import sys
import numpy as np
from tqdm import tqdm


def create_dataset_cli(argvs=sys.argv[1:]):
    import argparse

    from classifier_3D.utils.path import get_data_path
    from classifier_3D.utils.ply_file import read_ply, write_ply

    from range_net import (
        PATH_RANGE_NET,
        CITY_INFERANCE_FOLDER,
        PATH_INDEXES_TO_KEEP,
        PATH_SAMPLES,
        CITY_TO_KITTI_NUMBERS,
    )

    parser = argparse.ArgumentParser(
        "Pipeline to create samples for training RangeNet++."
    )
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
        "-ns",
        "--n_samples",
        type=int,
        default=10,
        help="The number of samples to draw, (default: 10).",
    )
    parser.add_argument(
        "-si",
        "--store_indexes",
        default=False,
        action="store_true",
        help="Whether to store selected indexes for reconstruction. (default: False)",
    )
    parser.add_argument(
        "-sp",
        "--store_ply",
        default=False,
        action="store_true",
        help="Whether to store samples in .ply format. (default: False)",
    )
    args = parser.parse_args(argvs)
    args = vars(args)
    print(args)

    # Clean paths
    path_folder_range_net_bins = (
        f"{PATH_RANGE_NET.replace('FOLDER', CITY_INFERANCE_FOLDER[args['file']])}"
    )
    if not os.path.exists(path_folder_range_net_bins):
        os.makedirs(path_folder_range_net_bins)
    else:
        [
            os.remove(f"{path_folder_range_net_bins}/{f}")
            for f in os.listdir(path_folder_range_net_bins)
            if f[-4:] == ".bin"
        ]

    path_folder_range_net_labels = path_folder_range_net_bins.replace(
        "velodyne", "labels"
    )
    if not os.path.exists(path_folder_range_net_labels):
        os.makedirs(path_folder_range_net_labels)
    else:
        [
            os.remove(f"{path_folder_range_net_labels}/{f}")
            for f in os.listdir(path_folder_range_net_labels)
            if f[-6:] == ".label"
        ]

    if args["store_indexes"]:
        path_store_indexes = f"{PATH_INDEXES_TO_KEEP}/{args['file']}"
        if not os.path.exists(path_store_indexes):
            os.mkdir(path_store_indexes)
        else:
            [
                os.remove(f"{path_store_indexes}/{f}")
                for f in os.listdir(path_store_indexes)
                if f[-4:] == ".npy"
            ]

    if args["store_ply"]:
        path_store_ply = f"{PATH_SAMPLES}/{args['file']}"
        if not os.path.exists(path_store_ply):
            os.mkdir(path_store_ply)
        else:
            [
                os.remove(f"{path_store_ply}/{f}")
                for f in os.listdir(path_store_ply)
                if f[-4:] == ".ply"
            ]

    # Load the dataset
    point_cloud_path = get_data_path(
        f"{args['file']}_with_features", args["is_train_data"]
    )
    point_cloud, _ = read_ply(point_cloud_path)
    points = np.vstack((point_cloud["x"], point_cloud["y"], point_cloud["z"])).T.astype(
        np.float32
    )
    if args["is_train_data"]:
        labels = np.array(
            [
                CITY_TO_KITTI_NUMBERS[label]
                for label in point_cloud["class"].astype(np.int32)
            ]
        )

    for n_sample in tqdm(range(args["n_samples"])):
        # Create a sample
        sample_points, indexes_to_keep = create_a_sample(args["file"], points)
        points_with_verticality = np.hstack(
            (
                sample_points,
                point_cloud["verticality"][indexes_to_keep].reshape(
                    (sample_points.shape[0], 1)
                ),
            )
        ).astype(np.float32)

        # Store the sample
        points_with_verticality.tofile(
            f"{path_folder_range_net_bins}/{str(n_sample).zfill(3)}.bin"
        )
        if args["is_train_data"]:
            labels[indexes_to_keep].astype(np.int32).tofile(
                f"{path_folder_range_net_labels}/{str(n_sample).zfill(3)}.label"
            )

        if args["store_indexes"]:
            # Store the indexes for reconstruction
            np.save(
                f"{path_store_indexes}/{str(n_sample).zfill(3)}.npy",
                indexes_to_keep.astype(bool),
            )

        if args["store_ply"]:
            write_ply(
                f"{path_store_ply}/{str(n_sample).zfill(3)}.ply",
                (points_with_verticality, labels[indexes_to_keep].astype(np.int32)),
                ["x", "y", "z", "verticality", "class"],
            )


def create_a_sample(
    point_cloud_name: str, points: np.ndarray
) -> [np.ndarray, np.ndarray]:
    from range_net import (
        CENTERS,
        Z_GROUNDS,
        Z_GROUND,
        ROTATIONS,
        MAX_DISTANCE,
        MAX_HEIGHT,
        MIN_DISTANCE,
        N_POINTS,
        BINS,
        N_PER_BINS,
    )

    # Pick a center
    centers_bound = CENTERS[point_cloud_name]

    weights = np.random.randint(0, 10, size=4).astype(float)
    weights /= weights.sum()

    center = np.append(
        weights @ centers_bound, [Z_GROUNDS[point_cloud_name] - Z_GROUND]
    )

    # Get the rotation
    theta = -ROTATIONS[point_cloud_name]

    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    # Transform the dataset
    centered_rotated_points = (points - np.expand_dims(center, axis=0)) @ rotation.T

    # Clip the points that are too far, too high, too close and sample the remaining points
    ranges = np.linalg.norm(centered_rotated_points, axis=1)
    ranges_x_y = np.linalg.norm(centered_rotated_points[:, 0:2], axis=1)
    geometrically_fitting = np.logical_and(
        np.logical_and(
            ranges < MAX_DISTANCE, centered_rotated_points[:, 2] < MAX_HEIGHT
        ),
        ranges_x_y > MIN_DISTANCE,
    )

    if geometrically_fitting.sum() > N_POINTS:
        to_keep = np.zeros(points.shape[0], dtype=bool)

        for idx_bin in range(len(BINS) - 1):
            indexes_in_bin = np.logical_and(
                np.logical_and(ranges > BINS[idx_bin], ranges < BINS[idx_bin + 1]),
                geometrically_fitting,
            )
            if indexes_in_bin.sum() < N_PER_BINS[idx_bin]:
                to_keep[indexes_in_bin] = True
            else:
                to_keep[
                    np.random.choice(
                        np.nonzero(indexes_in_bin)[0],
                        size=N_PER_BINS[idx_bin],
                        replace=False,
                    )
                ] = True
    else:
        to_keep = geometrically_fitting

    return (centered_rotated_points[to_keep], to_keep)
