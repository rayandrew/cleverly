import unittest

from numpy import ndarray, testing

from ..context import iris_data, iris_target, check_model_exist, purity_score, print_in_test
from cleverly.kmedoids.KMedoids import KMedoids


class KMedoidsTestSuite(unittest.TestCase):
    """KMedoids test cases."""

    @classmethod
    def setUpClass(self):
        self.filename = './tests/models/kmedoids.model'
        self.kmedoids = KMedoids(
            n_clusters=3)
        self.kmedoids.fit_predict(iris_data)

    def test_kmedois_return_labels_with_type_numpy_array(self):
        self.assertIsInstance(self.kmedoids.labels_, ndarray)
        print_in_test("KMedoids score: %f" %
                      purity_score(iris_target, self.kmedoids.labels_))


if __name__ == '__main__':
    unittest.main()
