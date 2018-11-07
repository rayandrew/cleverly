import numpy as np
import sys
from sklearn.base import BaseEstimator, ClassifierMixin

from ._base import *


BATCH_SIZE_ENUM = ['single', 'all']
ACTIVATION_FUNC_ENUM = ['identity', 'tanh', 'logistic', 'relu', 'softmax']
LOSS_FUNC_ENUM = ['squared_loss', 'log_loss', 'binary_log_loss']

BIAS_INPUT = 1


class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer=[3, ], batch_size=BATCH_SIZE_ENUM[0],
                 loss=LOSS_FUNC_ENUM[0], activation=ACTIVATION_FUNC_ENUM[2],
                 learning_rate=0.001, momentum=0.9, max_iter=200,
                 random_state=None, tol=1e-4, initial_weight=None):
        self.batch_size = batch_size  # one or all
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss = loss  # loss function
        self.activation = activation
        self.hidden_layer = hidden_layer
        self.random_state = random_state
        self.tol = tol
        self.momentum = momentum

        # Initiation of internal variable
        self.neural_net = []
        self.current_output = []
        self.error_terms = []
        self.error = sys.maxsize
        self.output_target = 0

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

        if isinstance(initial_weight, list):
            self.neural_net = initial_weight
            self.is_initial_weight = True
        else:
            # random weight
            for i in range(1, len(self.hidden_layer)):
                self.neural_net.append(np.random.randn(
                    self.hidden_layer[i-1] + 1, self.hidden_layer[i]))
            self.neural_net.append(
                np.random.randn(self.hidden_layer[i] + 1, 1))
            self.is_initial_weight = False

    @property
    def weights(self):
        return self.neural_net

    def init_zero_weights(self):
        weights = [] * len(self.neural_net)
        for i in self.neural_net:
            current_weight = []
            for j in i:
                current_weight.append(np.zeros(len(j)))
            weights.append(current_weight)

        return np.array(weights)

    def forward(self, X):
        input_data = np.append(X, [BIAS_INPUT])
        for i in range(len(self.neural_net)):
            net = np.dot(input_data, self.neural_net[i])
            output = self.activation_func(net)
            self.current_output.append(output)
            if i != (len(self.neural_net) - 1):
                # add bias to next layer
                output = np.append(output, [BIAS_INPUT])
            input_data = output

        self.output_target = input_data
        return input_data

    def backward(self, X, target):
        output = self.output_target
        # Calculate error
        self.error_terms = []  # reset error terms
        self.error = self.loss_func(target, output)
        # Append output layer error
        self.error_terms.append(output * (1 - output) * (target - output))
        layer = 0
        # Count error for hidden layer
        for h_layer in range(len(self.weights) - 1, 0, -1):
            error_hidden = []
            # Output before current layer
            for i in range(len(self.current_output[h_layer - 1])):
                error_sum = 0.0
                for j in range(len(self.weights[h_layer][i])):
                    error_sum += self.weights[h_layer][i][j] * \
                        self.error_terms[layer][j]

                err_temp = self.current_output[h_layer-1][i] * \
                    (1 - self.current_output[h_layer-1][i]) * \
                    error_sum

                error_hidden.append(err_temp)

            self.error_terms.append(error_hidden)

            # Update weights
            for i in range(len(self.error_terms[layer])):
                for j in range(len(self.current_output[h_layer-1])):
                    delta = self.learning_rate * \
                        self.error_terms[layer][i] * \
                        self.current_output[h_layer - 1][j] + \
                        self.momentum * self.delta_weights[h_layer][j][i]

                    self.delta_weights[h_layer][j][i] = delta
                    if self.batch_size == BATCH_SIZE_ENUM[0]:
                        self.weights[h_layer][j][i] += delta
                    elif self.batch_size == BATCH_SIZE_ENUM[1]:
                        self.accumulative_weight[h_layer][j][i] += delta

            # Update Bias
            for i in range(len(self.error_terms[layer])):
                delta = self.learning_rate * \
                    self.error_terms[layer][i] * \
                    BIAS_INPUT + \
                    self.momentum * \
                    self.delta_weights[h_layer][len(
                        self.delta_weights[h_layer]) - 1][i]

                self.delta_weights[h_layer][len(
                    self.delta_weights[h_layer]) - 1][i] = delta
                if self.batch_size == BATCH_SIZE_ENUM[0]:
                    self.weights[h_layer][len(
                        self.delta_weights[h_layer]) - 1][i] += delta
                elif self.batch_size == BATCH_SIZE_ENUM[1]:
                    self.accumulative_weight[h_layer][len(
                        self.delta_weights[h_layer]) - 1][i] += delta

            layer += 1

        # Update weight for input layer
        for i in range(len(self.error_terms[layer])):
            for j in range(len(X)):
                delta = self.learning_rate * \
                    self.error_terms[layer][i] * X[j] + \
                    self.momentum * self.delta_weights[0][j][i]

                self.delta_weights[0][j][i] = delta
                if self.batch_size == BATCH_SIZE_ENUM[0]:
                    self.weights[0][j][i] += delta  # Update weight immediately
                elif self.batch_size == BATCH_SIZE_ENUM[1]:
                    self.accumulative_weight[0][j][i] += delta

        # Update bias weight for input layer
        for i in range(len(self.error_terms[layer])):
            delta = self.learning_rate * \
                self.error_terms[layer][i] * BIAS_INPUT + \
                self.momentum * \
                self.delta_weights[0][len(self.delta_weights[0]) - 1][i]

            self.delta_weights[0][len(
                self.delta_weights[0]) - 1][i] = delta
            if self.batch_size == BATCH_SIZE_ENUM[0]:
                # Update weight immediately
                self.weights[0][len(self.delta_weights[0]) - 1][i] += delta
            elif self.batch_size == BATCH_SIZE_ENUM[1]:
                self.accumulative_weight[0][len(
                    self.delta_weights[0]) - 1][i] += delta

        # Empty saved outputs list
        self.current_output = []

    def update_weights(self):
        for i in range(len(self.neural_net)):
            self.neural_net[i] = np.add(
                self.neural_net[i], self.accumulative_weight[i])

        # Empty accumulative weights
        self.accumulative_weight = self.init_zero_weights()

    def fit(self, X, y):
        if len(X) <= 0:
            raise ValueError("X should have at least one row."
                             " %s was provided." % str(len(X)))

        X = np.array(X)
        # Insert input weight
        if not self.is_initial_weight:
            self.neural_net.insert(0, np.random.randn(
                len(X[0]) + 1, self.hidden_layer[0]))

        self.neural_net = np.array(self.neural_net)

        # Initiate delta weights
        self.delta_weights = self.init_zero_weights()

        # Initiate accumulative weights for batch size == all
        if self.batch_size == BATCH_SIZE_ENUM[1]:
            self.accumulative_weight = self.init_zero_weights()

        iter = 0
        while iter < self.max_iter and self.error > self.tol:
            for i in range(len(X)):
                self.forward(X[i])  # Feed forward
                self.backward(X[i], y[i])  # Back propagation

            if self.batch_size == BATCH_SIZE_ENUM[1]:
                self.update_weights()
            iter += 1

        return self

    def predict(self, X):
        return self.forward(X)
