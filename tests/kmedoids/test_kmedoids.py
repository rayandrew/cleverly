import unittest

from numpy import ndarray, testing
from sklearn.externals import joblib

from ..context import iris_data, iris_target, check_model_exist
from clustry.kmedoids.KMedoids import KMedoids


class KMedoidsTestSuite(unittest.TestCase):
    """KMedoids test cases."""

    @classmethod
    def setUpClass(self):
        self.filename = './tests/models/kmedoids.model'
        if check_model_exist(self.filename):
            self.kmedoids = joblib.load(self.filename)
        else:
            self.kmedoids = KMedoids(
                n_clusters=3)
            self.kmedoids.fit_predict(iris_data)
            joblib.dump(self.kmedoids, self.filename)

    def test_kmeans_return_labels_with_type_numpy_array(self):
        self.assertIsInstance(self.kmedoids.labels_, ndarray)


if __name__ == '__main__':
    unittest.main()
