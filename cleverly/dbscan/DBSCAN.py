#!/usr/bin/python
# -*- coding: utf-8 -*-
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np

class DBSCAN(BaseEstimator, ClusterMixin):
    OUTLIER = -1

    def __init__(self, minpts=2, eps=10):
        self.minpts = minpts
        self.eps = eps
        self.clusters = []
        self.labels_ = []

    def intersect(self, a, b):
        return len(list(set(a) & set(b))) > 0

    def compute_neighbors(self, distance_matrix):
        neighbors = []
        for i in range(len(distance_matrix)):
            neighbors_under_eps = []
            for neighbor in range(len(distance_matrix[i])):
                if distance_matrix[i][neighbor] <= self.eps \
                    and neighbor != i:
                    neighbors_under_eps.append(neighbor)
            neighbors.append(neighbors_under_eps)
        return neighbors

    def generate_clusters(self, neighbors_list):
        # initiate with the first data
        clusters = [neighbors_list[0] + [0]]
        for i in range(1, len(neighbors_list)):
            # for other data in the neighbour list
            # check if the data has an intersected cluster inside the result list
            # merge the list and append it to the result
            list_of_intersected_cluster = []
            new_cluster = neighbors_list[i] + [i]
            for cluster_num in range(len(clusters)):
                if self.intersect(neighbors_list[i],
                                  clusters[cluster_num]):
                    list_of_intersected_cluster.append(clusters[cluster_num])
                    new_cluster = new_cluster + clusters[cluster_num]

            # if the data is a new cluster / no intersected clusters
            if not list_of_intersected_cluster:
                clusters.append(neighbors_list[i] + [i])
            else:
                clusters.append(list(set(new_cluster)))
                # delete the merged clusters
                for old_cluster in list_of_intersected_cluster:
                    clusters.remove(old_cluster)
        return clusters

    def labelling(self, data, clusters):
        cluster_labels = [self.OUTLIER] * len(data)
        for i in range(len(self.clusters)):
            for j in range(len(self.clusters[i])):
                cluster_labels[self.clusters[i][j]] = i
        return cluster_labels

    def fit(self, X):
        distance_matrix = squareform(pdist(X))
        # compute the neighbors
        neighbors = self.compute_neighbors(distance_matrix)
        # clustering
        self.clusters = self.generate_clusters(neighbors)
        # filter out clusters with neighbors < minpts
        self.clusters = list(filter(lambda x: len(x) >= self.minpts,
                             self.clusters))
        # labelling
        self.labels_ = np.array(self.labelling(X, self.clusters))
        
        return self