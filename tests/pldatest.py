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
        enroledata = np.random.rand(100,10)
        enrolelabels = np.array([i%10 for i in range(enroledata.shape[0])])
        transformed = self.pldamodule.transform(enroledata,enrolelabels)
	testdata = np.random.rand(100,10)
	testlabels=np.arange(100)

	transformedtest = self.pldamodule.transform(testdata,testlabels)

        self.assertEqual(len(transformedtest.keys()),100)
        self.assertEqual(len(transformed.keys()),10)

        bkgdata = np.random.rand(5,10)
        self.pldamodule.norm(bkgdata,transformed)
        # random input
        for model,modelvec in transformed.iteritems():
            for testuttname,testvec in transformedtest.iteritems():
                score=self.pldamodule.score(model,modelvec,testvec)
                # Should be around this range
                self.assertTrue( -100 <= score <= 100)



if __name__ == '__main__':
    unittest.main()
