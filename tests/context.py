# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from sklearn import datasets, metrics
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


logging.basicConfig(format='%(message)s')

iris = datasets.load_iris()
iris_data = np.array(iris.data)
iris_target = iris.target


def check_model_exist(path):
    try:
        file = Path(path)
        return file.is_file()
    except:
        return False


def purity_score(y_true, y_pred):
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def print_in_test(string):
    logging.warning(string)


np.set_printoptions(threshold=np.inf)
