import numpy as np
import logging as logger
from app.PointCloudReader import PointCloudReader
from app.structured_clustering import cluster_by_structure
"""
Reads a roughly 2 sq-km of a subsampled LAS tile and classifies using structural clustering and produces final tile with
classification data. Classification classes are added in "user_data" field to make sure we can toggle and compare it 
with the raw classification data stored in "classification" field. 
"""

# pcr = PointCloudReader('./resources/C_37EZ1-eighth_2_0.las')  # read
# inlier_points, outlier_points = pcr.segment_ground_plane()
# pcr.point_cloud.points = inlier_points
# pcr.point_cloud.write('./resources/inliers_2_0.las')
# pcr.point_cloud.points = outlier_points
# pcr.point_cloud.write('./resources/outliers_2_0.las')
#
# first_returns = pcr.get_first_returns()
# last_returns = pcr.get_last_returns()
# remaining_returns = pcr.get_difference_returns(first_returns, last_returns)
#
# fr_points = pcr.point_cloud.points[first_returns]
# all_copy = pcr.point_cloud.points.copy()
# pcr.point_cloud.points = fr_points
# pcr.point_cloud.write('./resources/outliers_2_0_fr.las')
# pcr.point_cloud.points = all_copy
# lr_points = pcr.point_cloud.points[last_returns]
# pcr.point_cloud.points = lr_points
# pcr.point_cloud.write('./resources/outliers_2_0_lr.las')
# pcr.point_cloud.points = all_copy
# rm_points = pcr.point_cloud.points[remaining_returns]
# pcr.point_cloud.points = rm_points
# pcr.point_cloud.write('./resources/outliers_2_0_rem.las')
# pc = pcr.point_cloud
#
# label_index_map = cluster_by_structure(pc)
# pc.user_data = np.full(len(pc.user_data), fill_value=0)
# for new_label, label_index in label_index_map.items():
#     pcr.point_cloud.user_data[label_index] = new_label+1
# pcr.point_cloud.write('./resources/outlier2_2_0_vegetation_sc.las')
# logger.info("Vegetation separated.")
#
#
# tall_short_tree_classified = np.where(pcr.point_cloud.user_data!=0)[0]
# uclassified_remaining = np.where(pcr.point_cloud.user_data==0)[0]
# all_copy = pcr.point_cloud.points.copy()
# tall_short_tree_points = pcr.point_cloud.points[tall_short_tree_classified]
# pcr.point_cloud.points = tall_short_tree_points
# pcr.point_cloud.write('outlier2_0_tall_short_trees.las')
# logger.info("Vegetation Classified.")
#
# pcr.point_cloud.points= all_copy
# unclassified_remaining_points = pcr.point_cloud.points[uclassified_remaining]
# pcr.point_cloud.points= unclassified_remaining_points
# label_index_map = cluster_by_structure(pcr.point_cloud, eps=5, min_samples=10)
# pcr.point_cloud.user_data = np.full(len(pcr.point_cloud.user_data), fill_value=0)
# for new_label, label_index in label_index_map.items():
#     pcr.point_cloud.user_data[label_index] = 1
# buildings_unclassified = np.where(pcr.point_cloud.classification == 6)[0]
# waterbody_unclassfied = np.where(pcr.point_cloud.classification == 9)[0]
# pcr.point_cloud.user_data[buildings_unclassified]  = 6 # adding back the raw classification data for buildings to improve overall output
# pcr.point_cloud.user_data[waterbody_unclassfied] = 9 # adding back the water body raw classification data to improve overall output
# pcr.point_cloud.write('outlier2_0_unclassified.las')
# logger.info("Unclustered Building Side Walls Seperated and classified")
#
# pcr.point_cloud.points = all_copy
# fr_pcr = PointCloudReader("./resources/outliers_2_0_fr.las")
# fr_pc = fr_pcr.point_cloud
# # fr_pc = PointCloudReader.append_point_clouds(fr_pcr, pcr)
# # fr_pc.write('outlier2_0_fr_unclassified.las')
# # fr_pcr = PointCloudReader("outlier2_0_unclassified.las")
# label_index_map = cluster_by_structure(fr_pc, eps=2.5, min_samples=12, k_means_k=2, std_z_weight=1)
# fr_pc.user_data = np.full(len(fr_pc.user_data), fill_value=0)
# for new_label, label_index in label_index_map.items():
#     fr_pc.user_data[label_index] = 1
# buildings_unclassified = np.where(fr_pc.classification == 6)[0]
# waterbody_unclassfied = np.where(fr_pc.classification == 9)[0]
# fr_pc.user_data[buildings_unclassified] = 6
# fr_pc.user_data[waterbody_unclassfied] = 9
# fr_pc.write('./resources/outlier2_0_fr_classified.las')
# logger.info("First Return Classified for Buildings, Trees, Water")
#
# logger.info("Classifying last return")
#
# lr_pcr = PointCloudReader("./resources/outliers_2_0_lr.las")
# lr_pc = lr_pcr.point_cloud
# label_index_map = cluster_by_structure(lr_pc, eps=3.5, min_samples=8, k_means_k=2, std_z_weight=1)
# lr_pc.user_data = np.full(len(lr_pc.user_data), fill_value=0)
# for new_label, label_index in label_index_map.items():
#     lr_pc.user_data[label_index] = 1
# buildings_unclassified = np.where(lr_pc.classification == 6)[0]
# waterbody_unclassfied = np.where(lr_pc.classification == 9)[0]
# lr_pc.user_data[buildings_unclassified] = 6
# lr_pc.user_data[waterbody_unclassfied] = 9
# lr_pc.write('./resources/outlier2_0_lr_classified.las')

def merge_all():
    logger.info("Merging all classifications and generating final file with classifications")
    trees_pcr = PointCloudReader("./resources/outlier2_0_tall_short_trees.las")
    unclassified_pcr = PointCloudReader('./resources/outlier2_0_unclassified.las')
    fr_classified_pcr = PointCloudReader("./resources/outlier2_0_fr_classified.las")
    lr_classified_pcr = PointCloudReader("./resources/outlier2_0_lr_classified.las")
    ground_pcr = PointCloudReader("./resources/inliers_2_0.las")
    ground_pcr.point_cloud.user_data= np.full(len(ground_pcr.point_cloud.user_data), fill_value=7)
    trees_and_uc_cloud = PointCloudReader.append_point_clouds(trees_pcr.point_cloud, unclassified_pcr.point_cloud)
    trees_uc_fr_cloud = PointCloudReader.append_point_clouds(trees_and_uc_cloud, fr_classified_pcr.point_cloud)
    all_with_ground_pcr = PointCloudReader.append_point_clouds(trees_uc_fr_cloud, ground_pcr.point_cloud)
    trees_pcr.point_cloud.points=all_with_ground_pcr.points
    trees_pcr.point_cloud.write("./resources/final_classified.las")

merge_all()