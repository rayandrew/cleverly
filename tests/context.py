# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn import datasets
from pathlib import Path

from clustry import sample

iris = datasets.load_iris()
iris_data = np.array(iris.data)
iris_target = iris.target


def check_model_exist(path):
    file = Path(path)
    return file.is_file()


np.set_printoptions(threshold=np.inf)
