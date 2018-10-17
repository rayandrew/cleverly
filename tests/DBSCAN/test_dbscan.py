import unittest

from numpy import ndarray, testing
from sklearn.externals import joblib

from ..context import iris_data, iris_target, check_model_exist
from clustry.DBSCAN.DBSCAN import DBSCAN


class DBSCANTestSuite(unittest.TestCase):
    """DBSCAN test cases."""

    @classmethod
    def setUpClass(self):
        print(1)
        self.filename = './tests/models/dbscan.model'
        if check_model_exist(self.filename):
            self.dbscan = joblib.load(self.filename)
        else:
            self.dbscan = DBSCAN(
                minpts=10, eps=.4)
            self.dbscan.fit_predict(iris_data)
            joblib.dump(self.dbscan, self.filename)

    def test_dbscan_return_labels_with_type_numpy_array(self):
        self.assertIsInstance(self.dbscan.labels_, ndarray)


if __name__ == '__main__':
    unittest.main()
