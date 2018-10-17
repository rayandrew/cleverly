import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.base import BaseEstimator, ClusterMixin

class KMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters = 8, max_iter=300, tol=0.001):
        self.clusters_centers_ = None
        self.labels_ = None
        self.inertia_ = 0 # Sum of Squared Error for Checking Convergence
        self.n_iter_ = 0
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def initializeCentroidData(self, data):
        centroidsIndex = np.random.randint(len(data), size=self.n_clusters)
        centroidData = []
        for i in centroidsIndex:
            centroidData.append(data[i])
        return centroidData
    
    def clusterAssignment(self, data):
        # Compute Distance Matrix
        distances = euclidean_distances(data, self.clusters_centers_)
        # Assign data to clusters
        clusters = []
        for i in range(len(data)):
            clusters.append(np.argmin(distances[i]))
        return clusters

    def moveCentroid(self, data):
        centroidData = []
        for cluster in range(self.n_clusters):
            centroid = []
            for j in range(len(data[0])): ## Kalo bisa diganti, jangan len(data[0]) untuk j nya
                sumCluster = 0
                countCluster = 0
                for i in range(len(data)):
                    if(self.labels_[i] == cluster):
                        sumCluster += X[i][j]
                        countCluster += 1
                centroid.append(sumCluster/countCluster)
            centroidData.append(centroid)
        return centroidData
    
    def countError(self, data):
        error = 0
        for cluster in range(self.n_clusters):
        clusterData = []
        for i in range(len(data)):
            if(self.labels_[i] == cluster):
            clusterData.append(X[i])
        # Use distance matrix
        distanceMatrix = euclidean_distances(clusterData, [self.clusters_centers_[cluster]])
        # distanceMatrix[i][0], 0 karena hanya cuma ada satu cluster center yang diitung jaraknya 
        for i in range(len(distanceMatrix)):
            error += distanceMatrix[i][0]
        return error

    def fit(self, X):
        # Initial Random Centroid Index and Data
        self.clusters_centers_ = initializeCentroidData(X)
        # Loop (Until max iteration or below tolerance)
        for i in range(self.max_iter):
            # Increment iterations
            self.n_iter_ += 1
            # Compute Distance Matrix and Assign Cluster
            self.labels_ = clusterAssignment(X)
            # Move Centroid
            self.clusters_centers_ = moveCentroid(X)
            # Check Konvergence (error <= tol)
            self.inertia_ = countError(X)
            if(self.inertia_ <= self.tol):
                break
        return self
    