import unittest

from numpy import ndarray, testing
from sklearn.externals import joblib

from ..context import iris_data, iris_target, check_model_exist
from clustry.agglomerative.Agglomerative import Agglomerative


class AgglomerativeSingleEuclideanTestSuite(unittest.TestCase):
    """Agglomerative Single with Euclidean Distance test cases."""

    @classmethod
    def setUpClass(self):
        self.filename = './tests/models/agglo-single_euclidean.model'
        if check_model_exist(self.filename):
            self.agg = joblib.load(self.filename)
        else:
            self.agg = Agglomerative(
                linkage="single", affinity="euclidean", n_clusters=3)
            self.agg.fit_predict(iris_data)
            joblib.dump(self.agg, self.filename)

    def test_agglo_return_labels_with_type_numpy_array(self):
        self.assertIsInstance(self.agg.labels_, ndarray)


if __name__ == '__main__':
    unittest.main()
