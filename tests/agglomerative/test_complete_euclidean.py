import unittest

from numpy import ndarray, testing
from sklearn.externals import joblib

from ..context import iris_data, iris_target, check_model_exist, purity_score, print_in_test
from cleverly.agglomerative.Agglomerative import Agglomerative


class AgglomerativeCompleteEuclideanTestSuite(unittest.TestCase):
    """Agglomerative Complete with Euclidean Distance test cases."""

    @classmethod
    def setUpClass(self):
        self.filename = './tests/models/agglo-complete_euclidean.model'
        if check_model_exist(self.filename):
            self.agg = joblib.load(self.filename)
        else:
            self.agg = Agglomerative(
                linkage="complete", affinity="euclidean", n_clusters=3)
            self.agg.fit_predict(iris_data)
            joblib.dump(self.agg, self.filename)

    def test_agglo_return_labels_with_type_numpy_array(self):
        self.assertIsInstance(self.agg.labels_, ndarray)
        print_in_test("Agglomerative (Distance=Euclidean, Linkage=Complete): %f" %
                      purity_score(iris_target, self.agg.labels_))


if __name__ == '__main__':
    unittest.main()
