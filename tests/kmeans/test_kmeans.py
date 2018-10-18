import unittest

from numpy import ndarray, testing
from sklearn.externals import joblib

from ..context import iris_data, iris_target, check_model_exist
from clustry.kmeans.KMeans import KMeans


class KMeansTestSuite(unittest.TestCase):
    """KMeans test cases."""

    @classmethod
    def setUpClass(self):
        self.filename = './tests/models/kmeans.model'
        if check_model_exist(self.filename):
            self.kmeans = joblib.load(self.filename)
        else:
            self.kmeans = KMeans(
                n_clusters=3, max_iter=100, tol=0.002)
            self.kmeans.fit_predict(iris_data)
            joblib.dump(self.kmeans, self.filename)

    def test_kmeans_return_labels_with_type_numpy_array(self):
        self.assertIsInstance(self.kmeans.labels_, ndarray)


if __name__ == '__main__':
    unittest.main()
