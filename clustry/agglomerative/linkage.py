import numpy as np
from sklearn.metrics import euclidean_distances


def single(X, Y, distance_function=euclidean_distances):
    minVal = 999
    for i in range(len(X)):
        for j in range(len(Y)):
            distance = distance_function(
                [X[i]], [Y[j]])

            if distance < minVal:
                minVal = distance
    return minVal


def complete(X, Y, distance_function=euclidean_distances):
    maxVal = 0
    for i in range(len(X)):
        for j in range(len(Y)):
            distance = distance_function(
                [X[i]], [Y[j]])

            if distance > maxVal:
                maxVal = distance
    return maxVal


def average(X, Y, distance_function=euclidean_distances):
    total_distance = 0

    lenX = len(X)
    lenY = len(Y)
    count_distance = lenX * lenY
    for i in range(lenX):
        for j in range(lenY):
            total_distance += distance_function(
                [X[i]], [Y[j]])

    return total_distance / count_distance


def average_group(X, Y, distance_function=euclidean_distances):
    matrix_x = np.matrix(X)
    matrix_y = np.matrix(Y)

    mean_x = matrix_x.mean(0)
    mean_y = matrix_y.mean(0)

    return distance_function(mean_x, mean_y)
