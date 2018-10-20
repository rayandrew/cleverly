# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn import datasets, metrics
from pathlib import Path

iris = datasets.load_iris()
iris_data = np.array(iris.data)
iris_target = iris.target


def check_model_exist(path):
    file = Path(path)
    return file.is_file()

def purity_score(y_true, y_pred):
    # compute contingency matrix 
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

np.set_printoptions(threshold=np.inf)
