def compute_index_neighbors(query_points, cloud_points, radius_or_k_neighbors, use_radius):
    from sklearn.neighbors import KDTree

    kd_tree = KDTree(cloud_points, leaf_size=cloud_points.shape[0] // 1000, metric="euclidean")

    if use_radius:
        idx_neighbors_queries = kd_tree.query_radius(query_points, radius_or_k_neighbors)
    else:
        idx_neighbors_queries = kd_tree.query(query_points, k=radius_or_k_neighbors, return_distance=False)

    return idx_neighbors_queries
