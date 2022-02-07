import numpy as np

from classifier_3D.feature_extraction.PCA import compute_local_PCA


def get_features(query_points, cloud_points, radius_or_k_neighbors, use_radius=True, **kwargs):
    # !! Warning !! eigenvalues are given in ascending order
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius_or_k_neighbors, use_radius)

    features = []
    features_headers = []

    if kwargs.get("verticality", False):
        features.append(
            2 * np.arcsin(np.abs(all_eigenvectors[:, :, 0] @ np.array([0, 0, 1]))) / np.pi
        )  # 2 arcsin(|<n, e_z>|) / pi
        features_headers.append("verticality")

    if kwargs.get("linearity", False):
        features.append(1 - all_eigenvalues[:, 1] / (all_eigenvalues[:, 2] + 1e-8))  # 1 - lambda_2 / lambda_1
        features_headers.append("linearity")


    if kwargs.get("planarity", False):
        features.append(
            (all_eigenvalues[:, 1] - all_eigenvalues[:, 0]) / (all_eigenvalues[:, 2] + 1e-8)
        )  # lambda_2 - lambda_3 / lambda_1
        features_headers.append("planarity")

    if kwargs.get("sphericity", False):
        features.append(all_eigenvalues[:, 0] / (all_eigenvalues[:, 2] + 1e-8))  # lambda_3 - lambda_1
        features_headers.append("sphericity")

    return np.hstack(features), features_headers
