import numpy as np
import laspy as lp
from app.PlanarRansac import PlanarRansac


class PointCloudReader:
    """
    General utility for reading, manipulating point cloud using laspy. Contains utility methods to segment ground plane,
    get first and last return point clouds.
    """

    def __init__(self, las_file):
        self.point_cloud = lp.read(las_file)

    @staticmethod
    def append_point_clouds(dest_cloud, src_cloud):
        """
        Static utility method to merge any 2 LasPy point clouds by copying over properties
        and appending the src_cloud to dest_cloud
        :param dest_cloud: src point cloud would be appended behind this dest point cloud
        :param src_cloud: src cloud to append behind dest point cloud
        :return: appended point cloud
        """
        dest_x = np.array(dest_cloud.x)
        src_x = np.array(src_cloud.x)
        concat_x = np.append(dest_x, src_x)
        dest_y = np.array(dest_cloud.y)
        src_y = np.array(src_cloud.y)
        concat_y = np.append(dest_y, src_y)
        dest_z = np.array(dest_cloud.z)
        src_z = np.array(src_cloud.z)
        concat_z = np.append(dest_z, src_z)
        dest_user_data = np.array(dest_cloud.user_data)
        src_user_data = np.array(src_cloud.user_data)
        concat_user_data = np.append(dest_user_data, src_user_data)
        dest_classification = np.array(dest_cloud.classification)
        src_classification = np.array(src_cloud.classification)
        concat_classification = np.append(dest_classification, src_classification)
        dest_intensity = np.array(dest_cloud.intensity)
        src_intensity = np.array(src_cloud.intensity)
        concat_intensity = np.append(dest_intensity, src_intensity)
        dest_return_number = np.array(dest_cloud.return_number)
        src_return_number = np.array(src_cloud.return_number)
        concat_return_number = np.append(dest_return_number, src_return_number)
        dest_cloud.x = concat_x
        dest_cloud.y = concat_y
        dest_cloud.z = concat_z
        dest_cloud.user_data = concat_user_data
        dest_cloud.classification = concat_classification
        dest_cloud.intensity = concat_intensity
        dest_cloud.return_number = concat_return_number
        return dest_cloud

    def segment_ground_plane(self):
        """
        Segments Ground Plane using custom Ransac Plane Fitting algorithm
        :return: returns a touple of inlier and outlier points being classified as ground and not ground
        """
        points = np.vstack((self.point_cloud.x, self.point_cloud.y, self.point_cloud.z)).transpose()
        planar_ransac = PlanarRansac(points)
        model, inliers, outliers = planar_ransac.segment_plane()
        inlier_points = self.point_cloud.points[inliers]
        ground_classified = np.where(self.point_cloud.classification == 2)
        inliers_ground = np.concatenate([ground_classified[0], inliers[~np.isin(inliers, ground_classified)]])
        outliers = outliers[np.isin(outliers, inliers_ground, invert=True)]
        outlier_points = self.point_cloud.points[outliers]
        return inlier_points, outlier_points

    def get_first_returns(self):
        """
        Separates the first return points and returns a subset pointcloud of those points
        :return: a first return point cloud object
        """
        first_returns = np.where(self.point_cloud.return_number == 1)[0]
        return first_returns

    def get_last_returns(self):
        """
        Separates the last return points and returns a subset pointcloud of those points
        :return: returns a last return point cloud
        """
        max_return = np.max(self.point_cloud.return_number)
        last_returns = np.where(self.point_cloud.return_number == max_return)[0]
        return last_returns

    def get_difference_returns(self, first_returns, last_returns):
        """
        Returns a difference point clouds identified by difference in return numbers present in first_returns
        and last_returns
        :param first_returns: numpy array of indices of points having return_number=1
        :param last_returns:  numpy array of indices of points having return_number=max
        :return: a point cloud object representing the points in cloud having return number between first and last return
        """
        remaining_returns_mask = np.ones(len(self.point_cloud.return_number), dtype=bool)
        remaining_returns_mask[np.concatenate([first_returns, last_returns])] = False
        remaining_returns = np.where(remaining_returns_mask == True)[0]
        return remaining_returns


