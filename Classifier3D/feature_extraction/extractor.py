import numpy as np

from Classifier3D.features_extraction.PCA import compute_local_PCA


def compute_features(query_points, cloud_points, radius_or_k_neighbors, use_radius=True, **kwargs):
    # !! Warning !! eigenvalues are given in ascending order
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius_or_k_neighbors, use_radius)

    features = []

    if kwargs.get("verticality", False):
        features.append(
            2 * np.arcsin(np.abs(all_eigenvectors[:, :, 0] @ np.array([0, 0, 1]))) / np.pi
        )  # 2 arcsin(|<n, e_z>|) / pi

    if kwargs.get("linearity", False):
        features.append(1 - all_eigenvalues[:, 1] / (all_eigenvalues[:, 2] + 1e-8))  # 1 - lambda_2 / lambda_1

    if kwargs.get("planarity", False):
        features.append(all_eigenvalues[:, 1] - all_eigenvalues[:, 0]) / (
            all_eigenvalues[:, 2] + 1e-8
        )  # lambda_2 - lambda_3 / lambda_1

    if kwargs.get("sphericity", False):
        features.append(all_eigenvalues[:, 0] / (all_eigenvalues[:, 2] + 1e-8))  # lambda_3 - lambda_1

    return np.hstack(features)
