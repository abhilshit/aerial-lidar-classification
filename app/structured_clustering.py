from app.PointCloudReader import PointCloudReader
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from math import sqrt
import logging as logger


def calculate_standard_deviation(cluster_points , std_z_weight=3, minmax_z_weight=2, add_min_max_z=False):
    """
    Calculate standard deviations and means of each DBSCAN cluster represented by the cluster points
    :param cluster_points: points in the DBSCAN cluster Numpy array of vertically stacked X,Y,Z,return_number,
     number_of_returns points.

    :param std_z_weight: weight to exponentially increase the effect of increase in Z standard deviation exponentially
    :param minmax_z_weight: weight to exponentially increase the effect of increase in maxZ-minZ value exponentially
    :return: list of standard deviations and mean values used to identify a structure of a DBscan cluster through K-Means
    """
    z_values = cluster_points[:, 2]
    y_values = cluster_points[:, 1]
    x_values = cluster_points[:, 0]
    standard_deviation_z = np.std(z_values)
    standard_deviation_z = standard_deviation_z ** std_z_weight
    standard_deviation_y = np.std(y_values)
    standard_deviation_y = sqrt(standard_deviation_y)
    standard_deviation_x = np.std(x_values)
    standard_deviation_x = sqrt(standard_deviation_x)
    standard_deviation_return_num = np.std(cluster_points[:, 3])
    # x_minmax_diff = x_values.max() - x_values.min() ** 2
    # y_minmax_diff = y_values.max() - y_values.min() ** 2
    # standard_deviation_num_returns = np.std(cluster_points[:, 4])
    # standard_deviation_intensity = standard_deviation_intensity ** 2
    #standard_deviation_return_num = standard_deviation_return_num**2
    mean_z = np.average(z_values)
    mean_z = (-1 * mean_z**2) if (mean_z < 0)  else mean_z**2
    values = [standard_deviation_z, standard_deviation_y, standard_deviation_x, standard_deviation_return_num, mean_z]
    if add_min_max_z:
        z_minmax_diff = (z_values.max() - z_values.min()) ** minmax_z_weight
        values.append(z_minmax_diff)
    return values


def norm(vector):
    """
    Calculates the normal of a vector
    :param vector: input vector
    :return: norm of the input vector
    """
    return sqrt(sum(x * x for x in vector))


def cosine_similarity(vec_a, vec_b):
    """
    Calculates the cosine similarity between given vectors
    :param vec_a: first vector
    :param vec_b: second vector
    :return: cosine similarity score (angle) between 2 input vectors
    """
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)

def calc_euclidean_distance(vec1, vec2):
    """
    Calculates Euclidean Distance between 2 vectors
    :param vec1: first vector
    :param vec2: second vector
    :return: euclidean distance between 2 input vectors
    """
    dist = (vec1 - vec2) ** 2
    dist = np.sqrt(np.sum(dist))
    return dist

def cluster_by_structure(pc, eps=2.5, min_samples=15, k_means_k = 3 , std_z_weight=3, add_min_max_z=False):
    """
     Implements Cluster by Structure algorithm which is a DBSCAN followed by extraction of structural features of DBSCAN
     clusters which are then clustered by k-means using those structural features.
    :param pc: point cloud to cluster
    :param eps: DBSCAN epsillon value
    :param min_samples: DBSCAN min_sample value
    :param k_means_k: k_means
    :return:
    """
    points = np.vstack((pc.x, pc.y, pc.z)).transpose()
    logger.info("Performing DBSCAN")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    dbscan.fit(points)
    labels = dbscan.labels_
    logger.info("DBSCAN done !")
    label_index_map = {v: np.where(labels == v)[0] for v in np.unique(labels)}
    all_cluster_deviations = []
    for class_, label_indices in label_index_map.items():
        if class_ == -1:
            continue
        if len(label_indices) > 10000:
            sampled_label_indices = np.random.choice(label_indices, size=10000)
        else:
            sampled_label_indices = label_indices
        cluster_points = np.vstack((np.array(pc.x)[sampled_label_indices],
                                    np.array(pc.y)[sampled_label_indices],
                                    np.array(pc.z)[sampled_label_indices],
                                    np.array(pc.return_number)[sampled_label_indices],
                                    np.array(pc.number_of_returns)[sampled_label_indices]
                                    , np.array(pc.intensity)[sampled_label_indices])).transpose()
        cluster_deviations = calculate_standard_deviation(cluster_points, std_z_weight, add_min_max_z)
        all_cluster_deviations.append(cluster_deviations)

    logger.info("Doing Kmeans")
    kmeans = KMeans(k_means_k)
    deviation_cluster_labels = kmeans.fit(np.array(all_cluster_deviations)).labels_
    logger.info("kmeans done")
    new_label_old_label_map = {v: np.where(deviation_cluster_labels == v)[0] for v in
                               np.unique(deviation_cluster_labels)}

    new_label_index_map = {}
    for new_label, label_index in new_label_old_label_map.items():
        new_label_indices = []
        for old_label in label_index:
            old_label_point_indices = label_index_map[old_label]
            new_label_indices.extend(old_label_point_indices)
        new_label_index_map[new_label] = new_label_indices
    return new_label_index_map

