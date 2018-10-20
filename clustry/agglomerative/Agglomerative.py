import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances

from .linkage import single, complete, average, average_group
from sklearn.externals.joblib import Parallel, delayed

LINKAGE_TYPE = ["complete", "single", "average", "average-group"]
AFFINITY_TYPE = ["euclidean", "manhattan"]


class Agglomerative(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, linkage=LINKAGE_TYPE[0], affinity=AFFINITY_TYPE[0]):
        self.clusters = []
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity

    def distance_function(self, i, j):
        return pairwise_distances(i, j, metric=self.affinity)[0][0]

    def cluster_dist(self, data, i, j):
        clust1 = data[i]
        clust2 = data[j]

        assert self.linkage in LINKAGE_TYPE

        if self.linkage == "single":
            return single(clust1, clust2, self.distance_function)
        elif self.linkage == "average":
            return average(clust1, clust2, self.distance_function)
        elif self.linkage == "average-group":
            return average_group(clust1, clust2, self.distance_function)
        else:
            return complete(clust1, clust2, self.distance_function)

    def distance_matrix(self, data):
        length = len(self.clusters)
        mat_distance = np.zeros((length, length))
        for i in range(len(mat_distance)):
            for j in range(len(mat_distance[i])):
                mat_distance[i][j] = self.cluster_dist(
                    data, self.clusters[i], self.clusters[j])
        return mat_distance

    def agglo_algo(self, X):
        lenX = len(X)
        # Initiate clusters
        for i in range(lenX):
            arr = [i]
            self.clusters.append(arr)

        # Initiate labels
        self.labels_ = np.zeros((lenX,))

        # Main loop
        while len(self.clusters) > self.n_clusters:
            dist_matrix = self.distance_matrix(X)

            # Search min in distance matrix
            minVal = 999
            # print("hello", len(dist_matrix[i]))
            for i in range(0, len(dist_matrix)):
                for j in range(i, len(dist_matrix[i])):
                    # print(i, j, dist_matrix[i][j])
                    if dist_matrix[i][j] < minVal and dist_matrix[i][j] > 0 and i != j:
                        # print(i, j)
                        index = (i, j)
                        minVal = dist_matrix[i][j]

            # print("INDEX", index)
            # Merge cluster and delete the merged cluster
            for i in range(len(self.clusters[index[1]])):
                self.clusters[index[0]].append(self.clusters[index[1]][i])
            del self.clusters[index[1]]

        # Changed clustering result into labels array
        for i in range(len(self.clusters)):
            for j in range(len(self.clusters[i])):
                self.labels_[self.clusters[i][j]] = i

        self.labels_ = self.labels_.astype(int)
        return self.labels_

    def fit(self, X):
        if self.n_clusters <= 0:
            raise ValueError("n_clusters should be an integer greater than 0."
                             " %s was provided." % str(self.n_clusters))

        if self.linkage not in LINKAGE_TYPE:
            raise ValueError("Unknown linkage type %s. "
                             "Valid options are %s" % (self.linkage,
                                                       LINKAGE_TYPE))

        if isinstance(self.affinity, (str, bytes)) and self.affinity not in AFFINITY_TYPE:
            raise ValueError("Unknown affinity type %s. "
                             "Valid options are %s" % (self.affinity,
                                                       AFFINITY_TYPE))

        self.agglo_algo(X)

        return self
