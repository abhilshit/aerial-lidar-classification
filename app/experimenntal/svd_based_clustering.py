from app.PointCloudReader import PointCloudReader
import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering
from multiprocessing.dummy import Pool
import scipy
from itertools import repeat
from math import sqrt
from sklearn.cluster import AgglomerativeClustering


def calculate_svd(cluster_points):
    """
    Performs Singular Value decomposition of the data points
    :param cluster_points: data points
    :return: Principal Directions (V) * Sigma (S) indicating principal directions with magnitude
    """
    u, s, vt = scipy.linalg.svd(cluster_points)
    if s.shape[0] < vt.shape[0]:
        difference = vt.shape[0] - s.shape[0]
        for i in range(len(s),len(s)+difference):
            s = np.append(s,0.0)
    principal_directions_with_magnitude = s*vt.transpose()
    return principal_directions_with_magnitude


def norm(vector):
    return sqrt(sum(x * x for x in vector))


def cosine_similarity(vec_a, vec_b):
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)

def calc_euclidean_distance(vec1, vec2):
    dist = (vec1 - vec2) ** 2
    dist = np.sqrt(np.sum(dist))
    return dist


def calc_similarity(args):
    item, label, cluster_sim_scores = args
    other_label, other_principal_directions = item
    if label == other_label:
        cluster_sim_scores[label][other_label] = 0
    else:
        first_sim_score = calc_euclidean_distance(principal_directions[0], other_principal_directions[0])
        #second_sim_score = calc_euclidean_distance(principal_directions[1], other_principal_directions[1])
        #third_sim_score = calc_euclidean_distance(principal_directions[2], other_principal_directions[2])
        weighted_sim_score = first_sim_score/1000
        cluster_sim_scores[label][other_label] = weighted_sim_score


if __name__ == '__main__':
    pcr = PointCloudReader("../../resources/outliers_2_0_rem.las")
    pc = pcr.point_cloud
    pc
    points = np.vstack((pc.x, pc.y, pc.z)).transpose()
    dbscan = DBSCAN(eps=2, min_samples=10, metric='euclidean').fit(points)
    labels = dbscan.labels_
    print(np.unique(labels))
    label_index_map = {i: (labels == i).nonzero()[0] for i in np.unique(labels)}
    pcr.point_cloud.user_data = labels
    pcr.point_cloud.write('outlier2_2_0.las')

    class_principal_direction_map = {}
    # points_copy = pcr.point_cloud.points.copy()
    for class_, label_indices in label_index_map.items():
        if class_ == -1:
            continue
        if len(label_indices) > 5000:
            sampled_label_indices = np.random.choice(label_indices, size=5000)
        else:
            sampled_label_indices = label_indices
        cluster_points = np.vstack((np.array(pcr.point_cloud.x)[sampled_label_indices],
                                    np.array(pcr.point_cloud.y)[sampled_label_indices],
                                    np.array(pcr.point_cloud.z)[sampled_label_indices])).transpose()
        cluster_principal_directions = calculate_svd(cluster_points)
        class_principal_direction_map[class_] = cluster_principal_directions

    similar_labels = {}
    cluster_sim_scores = np.full((len(label_index_map.keys()) - 1, len(label_index_map.keys()) - 1), 0, dtype="float64")
    pool = Pool(processes=4)
    for label, principal_directions in class_principal_direction_map.items():
        pool.map(calc_similarity,
                 zip(class_principal_direction_map.items(), repeat(label), repeat(cluster_sim_scores)))
        print("Calculating Similarity Matrix for label : " + str(label) + "/" + str(len(np.unique(labels))))

    ag = AgglomerativeClustering(n_clusters=2)
    new_clusters = ag.fit(cluster_sim_scores).labels_
    new_label_old_label_map = {i: (new_clusters == i).nonzero()[0] for i in np.unique(new_clusters)}
    pcr.point_cloud.user_data = np.full(len(pcr.point_cloud.user_data), fill_value=0)
    for new_label, label_index in new_label_old_label_map.items():
        for old_label in label_index:
            old_label_point_indices = label_index_map[old_label]
            pcr.point_cloud.user_data[old_label_point_indices] = new_label+1

    pcr.point_cloud.write('../../resources/svd_outlier2_2_0_veg_buildings.las')
    print("done")