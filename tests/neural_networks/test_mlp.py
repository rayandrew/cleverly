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
            self.mlp = None
            # self.kmeans.fit_predict(iris_data)
            # joblib.dump(self.kmeans, self.filename)

    # def test_kmeans_return_labels_with_type_numpy_array(self):
    #     self.assertIsInstance(self.kmeans.labels_, ndarray)
    #     print_in_test("KMeans (max_iter=300, tol=0.002): %f" %
    #                   purity_score(iris_target, self.kmeans.labels_))


if __name__ == '__main__':
    unittest.main()
