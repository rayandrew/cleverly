import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.base import BaseEstimator, ClusterMixin
from random import choice
from copy import copy

class KMedoids(BaseEstimator, ClusterMixin):
    MAX_RANDOM = 300

    def __init__(self, n_clusters = 8, max_iter=300, tol=0.001):
        self.clusters_centers_ = None
        self.labels_ = None
        self.inertia_ = 0 # Sum of Error for Checking Convergence
        self.n_iter_ = 0
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.medoids_history = []
        self.current_centroids = None

    def initializeCentroidData(self, data):
        centroidsIndex = np.random.randint(len(data), size=self.n_clusters)
        centroidData = []
        for i in centroidsIndex:
            centroidData.append(data[i])
        # Save centroid
        self.medoids_history.append(list(centroidsIndex))
        self.current_centroids = centroidsIndex
        return centroidData
    
    def clusterAssignment(self, data):
        # Compute Distance Matrix
        distances = manhattan_distances(data, self.clusters_centers_)
        # Assign data to clusters
        clusters = []
        for i in range(len(data)):
            clusters.append(np.argmin(distances[i]))
        return clusters

    def moveCentroid(self, data):
        clusters = [[] for i in range(self.n_clusters)]
        for i in range(len(self.labels_)):
            clusters[self.labels_[i]].append(i)
        # Search for new medoid configuration
        stop = False
        i = 0
        while not stop and i < self.MAX_RANDOM:
            # Get a random cluster
            random_cluster = np.random.randint(0, self.n_clusters)
            # Randomly pick a member of that cluster to be the next medoid
            medoid = choice(clusters[random_cluster])
            centroidsIndex = []
            for i in range(self.n_clusters):
                if i != random_cluster:
                    centroidsIndex.append(self.current_centroids[i])
                else:
                    centroidsIndex.append(medoid)
            if not centroidsIndex in self.medoids_history:
                self.medoids_history.append(centroidsIndex)
                stop = True
            else:
                i += 1
        # didn't found new configuration
        if not stop:
            return self.clusters_centers_

        self.current_centroids = centroidsIndex
        centroidData = []
        for i in centroidsIndex:
            centroidData.append(data[i])
        return centroidData

    def countError(self, data):
        error = 0
        for cluster in range(self.n_clusters):
            clusterData = []
            for i in range(len(data)):
                if(self.labels_[i] == cluster):
                    clusterData.append(data[i])
            distanceMatrix = manhattan_distances(clusterData, [self.clusters_centers_[cluster]])
            # distanceMatrix[i][0], 0 karena hanya cuma ada satu cluster center yang diitung jaraknya 
            for i in range(len(distanceMatrix)):
                error += distanceMatrix[i][0]
        return error

    def fit(self, X):
        # Initial Random Centroid Index and Data
        self.clusters_centers_ = self.initializeCentroidData(X)
        clusters_centers = copy(self.clusters_centers_)
        # Compute Distance Matrix and Assign Cluster
        self.labels_ = copy(self.clusterAssignment(X))
        labels = self.labels_
        # initiate error
        self.inertia_ = self.countError(X)
        minimum_error = self.inertia_

        # Loop (Until max iteration or below tolerance)
        for i in range(self.max_iter):
            # Increment iterations
            self.n_iter_ += 1
            # Move Centroid
            self.clusters_centers_ = self.moveCentroid(X)
            # Compute Distance Matrix and Assign Cluster
            self.labels_ = self.clusterAssignment(X)
            # Check error
            self.inertia_ = self.countError(X)
            new_error = self.inertia_
            if new_error < minimum_error:
                clusters_centers = copy(self.clusters_centers_)
                labels = copy(self.labels_)
                minimum_error = new_error
            else:
                # Revert back
                self.clusters_centers_ = copy(clusters_centers)
                self.labels_ = copy(labels)
                self.inertia_ = minimum_error

            if(self.inertia_ <= self.tol):
                break
        return self