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
        self.activation = activation
        self.hidden_layer = hidden_layer
        self.random_state = random_state
        self.tol = tol
        self.momentum = momentum
        self.neural_net = []
        self.current_output = []

        if self.loss == LOSS_FUNC_ENUM[0]:
            self.loss_func = squared_loss
        elif self.loss == LOSS_FUNC_ENUM[1]:
            self.loss_func = log_loss
        elif self.loss == LOSS_FUNC_ENUM[2]:
            self.loss_func = binary_log_loss
        else:
            raise ValueError("Loss function is not defined.")

        if self.activation == ACTIVATION_FUNC_ENUM[0]:
            self.activation_func = identity
        elif self.activation == ACTIVATION_FUNC_ENUM[1]:
            self.activation_func = tanh
        elif self.activation == ACTIVATION_FUNC_ENUM[2]:
            self.activation_func = logistic
        elif self.activation == ACTIVATION_FUNC_ENUM[3]:
            self.activation_func = relu
        elif self.activation == ACTIVATION_FUNC_ENUM[4]:
            self.activation_func = softmax
        else:
            raise ValueError("Activation function is not defined.")
        
        # random weight
        for i in range(1, len(self.hidden_layer)):
            self.neural_net.append(np.random.randn(self.hidden_layer[i-1], self.hidden_layer[i]))
        self.neural_net.append(np.random.randn(self.hidden_layer[i], 1))

    @property
    def weight(self):
        return self.neural_net

    def forward(self, X):
        input_data = X
        for i in range(len(self.neural_net)):
            net = np.dot(input_data, self.neural_net[i])
            output = self.activation_func(net)
            input_data = output
            self.current_output.append(output)

    def fit(self, X, y):
        if len(X) <= 0:
            raise ValueError("X should have at least one row."
                             " %s was provided." % str(len(X)))
        # Insert input weight
        self.neural_net.insert(0, np.random.randn(len(X[0]), self.hidden_layer[0]))
        # Feed forward
        for i in range(len(X)):
            self.forward(X[i])
            print(self.current_output)
            self.current_output = []

        return self

    def predict(self, X):
        pass
