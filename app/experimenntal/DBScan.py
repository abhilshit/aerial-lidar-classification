import numpy as np
import logging as logger
class DBScan:
    """
    DBSCAN implementation
    """
    def __init__(self, data, epsillon=2, minPts=100):
        self.data = data
        self.point_count = len(data)
        self.labels = np.full((self.point_count,),0) #initialize default labels with zeros
        self.eps = epsillon
        self.minPts = minPts


    def __get_neighbours(self,point_index):
        """
        Gets neighbours of point through distance calculation using numpy vectorization
        :param point_index:
        :return:
        """
        # neighbors = []
        dist = (self.data - self.data[point_index]) ** 2
        dist = np.sum(dist, axis=1)
        neighbors = np.where(np.sqrt(dist)<self.eps)

        return neighbors[0]

    def __expand_cluster(self, point_index, neighbours, current_class_id):
        """
        Expands cluster using the core-point, expansion logic.
        :param point_index:
        :param neighbours:
        :param current_class_id:
        :return:
        """
        # expand a new cluster witl label current_class_id with the inital point represented by point_index
        self.labels[point_index] = current_class_id


        #iterate over each neighbour of point at point_index
        i = 0
        while i < len(neighbours):
            ith_neighbour=neighbours[i]
            # if it was labelled as -1 during inital point search, correct the label, if it is unlabelled i.e 0 apply the label
            if self.labels[ith_neighbour] < 0 :
                self.labels[ith_neighbour] = current_class_id

            #if it was unlabelled, find all neighours of this point
            if self.labels[ith_neighbour] == 0:
                # Find all the neighbors of Pn
                ith_neighbour_neighbours = self.__get_neighbours(ith_neighbour)
                if len(ith_neighbour_neighbours) >= self.minPts:
                    neighbours = np.concatenate((neighbours,ith_neighbour_neighbours))
            i=i+1


    def fit(self):
        """
        Applies clustering to the data
        :return: labels of the data
        """
        current_class_id = 0  # class id of the current cluster
        for point_index in range(0, self.point_count):
            logger("Processing point: "+ str(point_index) + "/" + str(self.point_count))
            # continue if label is already assigned to a point
            if not self.labels[point_index]==0:
                continue
            #Find all qualifying neighbours in epsilon distance range
            neighbours = self.__get_neighbours(point_index)
            if len(neighbours)<self.minPts:
                self.labels[point_index] = -1 #if no. of neighbours is less than minpoints, label this point as outlier
            else:
                current_class_id=current_class_id+1
                self.__expand_cluster(point_index, neighbours, current_class_id)

        return self.labels


