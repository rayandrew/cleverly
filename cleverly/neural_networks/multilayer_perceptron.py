import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import manhattan_distances

from sklearn.base import BaseEstimator, ClassifierMixin

from ._base import *

BATCH_SIZE_ENUM = ['single', 'all']
ACTIVATION_FUNC_ENUM = ['identity', 'tanh', 'logistic', 'relu', 'softmax']
LOSS_FUNC_ENUM = ['squared_loss', 'log_loss', 'binary_log_loss']


class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer=[3, ], alpha=0.0001,
                 batch_size=BATCH_SIZE_ENUM[0], loss=LOSS_FUNC_ENUM[0], activation=ACTIVATION_FUNC_ENUM[0],
                 learning_rate=0.001, momentum=0.9, max_iter=200,
                 random_state=None, tol=1e-4):
        self.batch_size = batch_size  # one or all
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss = loss  # loss function
        self.hidden_layer = hidden_layer
        self.random_state = random_state
        self.tol = tol
        self.momentum = momentum

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
