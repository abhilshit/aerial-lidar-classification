import numpy as np
import random

class PlanarRansac:
    """
    Implementation of Planar RANSAC (Random Sample Consensus)
    Finds the best planar model using RANSAC and also returns the best inliers according to threshold defined.
    """
    def __init__(self, pts, thresh=0.20, minPoints=80, maxIteration=2000):
        self.inliers = []
        self.outliers = []
        self.planar_model = []
        self.pts = pts
        self.thresh = thresh
        self.minPoints = minPoints
        self. maxIteration = maxIteration

    def segment_plane(self):
        """
        Finds the best planar model using RANSAC and also returns the best inliers according to threshold defined.
        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.planar_model`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers
        ---
        """
        n_points = self.pts.shape[0]
        best_eq = []
        best_inliers = []
        all_indices = np.arange(start=0, stop=n_points)
        for it in range(self.maxIteration):

            # Samples 3 random points
            sample_idx = random.sample(range(1, n_points - 1), 3)
            pt_samples = self.pts[sample_idx]

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to a plane
            dist_pt = (plane_eq[0] * self.pts[:, 0] + plane_eq[1] * self.pts[:, 1] + plane_eq[2] * self.pts[:, 2] + plane_eq[
                3]) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= self.thresh)[0]
            if (len(pt_id_inliers) > len(best_inliers)):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            self.inliers = best_inliers
            self.planar_model = best_eq
        self.outliers = all_indices[np.isin(all_indices,self.inliers, invert=True)]
        return self.planar_model, self.inliers, self.outliers
