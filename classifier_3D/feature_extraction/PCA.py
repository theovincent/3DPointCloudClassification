from tqdm import tqdm
from sklearn.neighbors import KDTree

import numpy as np


def PCA(points):
    mean_points = np.mean(points, axis=0)
    centered_points = points - mean_points
    conv_matrix = centered_points.T @ centered_points / len(centered_points)

    eigenvalues, eigenvectors = np.linalg.eigh(conv_matrix)

    return eigenvalues, eigenvectors


def compute_local_PCA(query_points, cloud_points, radius_or_k_neighbors, use_radius):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    kd_tree = KDTree(cloud_points, leaf_size=4, metric="euclidean")

    if use_radius:
        idx_neighbors_queries = kd_tree.query_radius(query_points, radius_or_k_neighbors)
    else:
        idx_neighbors_queries = kd_tree.query(query_points, k=radius_or_k_neighbors, return_distance=False)

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    for idx_query, idx_neighbors in tqdm(enumerate(idx_neighbors_queries)):
        all_eigenvalues[idx_query], all_eigenvectors[idx_query] = PCA(cloud_points[idx_neighbors])

    return all_eigenvalues, all_eigenvectors
