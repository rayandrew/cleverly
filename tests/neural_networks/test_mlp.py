import unittest

from numpy import ndarray, testing
from sklearn.externals import joblib

from ..context import iris_data, iris_target, check_model_exist, purity_score, print_in_test
from cleverly.neural_networks.multilayer_perceptron import MLP


class MLPTestSuite(unittest.TestCase):
    """MLP test cases."""

    @classmethod
    def setUpClass(self):
        self.filename = './tests/models/mlp.model'
        if check_model_exist(self.filename):
            self.mlp = joblib.load(self.filename)
        else:
            self.mlp = MLP(
                hidden_layer=[3, 2],
                learning_rate=0.2,
                momentum=0.9,
                batch_size='all',
                max_iter=2,
                initial_weight=[
                    [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 1, 1]],
                    [[1, 1], [1, 1], [1, 1], [1, 1]],
                    [[1], [1], [1]]
                ])
            self.mlp.fit([[2, 5], [3, 4], [-2, -1]], [1, 1, 0])
            joblib.dump(self.mlp, self.filename)

    def test_mlp_return_labels_with_type_numpy_array(self):
        pred = self.mlp.predict([[4, 8]])
        print_in_test("Prediction of [4, 8]: %f" % pred)
        self.assertIsInstance(pred, ndarray)


if __name__ == '__main__':
    unittest.main()
