from tqdm import tqdm

import numpy as np


def PCA(points):
    mean_points = np.mean(points, axis=0)
    centered_points = points - mean_points
    conv_matrix = centered_points.T @ centered_points / len(centered_points)

    eigenvalues, eigenvectors = np.linalg.eigh(conv_matrix)

    return eigenvalues, eigenvectors


def compute_local_PCA(query_points, cloud_points, radius_or_k_neighbors, use_radius):
    from classifier_3D.utils.neighbors import compute_index_neighbors

    idx_neighbors_queries = compute_index_neighbors(query_points, cloud_points, radius_or_k_neighbors, use_radius)

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    for idx_query, idx_neighbors in tqdm(enumerate(idx_neighbors_queries)):
        all_eigenvalues[idx_query], all_eigenvectors[idx_query] = PCA(cloud_points[idx_neighbors])

    return all_eigenvalues, all_eigenvectors
