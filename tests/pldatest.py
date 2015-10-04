import unittest
import numpy as np
from liblda import PLDA

class PLDATest(unittest.TestCase):
    def setUp(self):
        self.pldamodule = PLDA()
        self.data = np.random.rand(2000,10)
        self.labels = np.array([i%10 for i in range(self.data.shape[0])])
    def test_fittransform(self):
        self.assertIsNone(self.pldamodule.fit(self.data,self.labels))
        testdata = np.random.rand(100,10)
        testlabels = np.array([i%10 for i in range(testdata.shape[0])])
        transformed = self.pldamodule.transform(testdata,testlabels)
        self.assertEqual(len(transformed.keys()),10)

        bkgdata = np.random.rand(5,10)
        self.pldamodule.norm(bkgdata,transformed)
        score = self.pldamodule.score(transformed.keys()[0],transformed.values()[0],transformed.values()[2])
        # Should be around this range
        self.assertTrue( -100 <= score <= 100)


if __name__ == '__main__':
    unittest.main()
