import unittest
import numpy as np
from liblda import PLDA


class PLDATest(unittest.TestCase):

    def setUp(self):
        self.pldamodule = PLDA()
        self.data = np.random.rand(2000, 10)
        self.labels = np.array([i % 10 for i in range(self.data.shape[0])])

    def test_fittransform(self):
        self.assertIsNone(self.pldamodule.fit(self.data, self.labels))
        enroledata = np.random.rand(100, 10)
        enrolelabels = np.array([i % 10 for i in range(enroledata.shape[0])])
        transformed = self.pldamodule.transform(enroledata, enrolelabels)
        testdata = np.random.rand(100, 10)
        testlabels = np.arange(100)

        transformedtest = self.pldamodule.transform(testdata, testlabels)

        self.assertEqual(len(transformedtest.keys()), 100)
        self.assertEqual(len(transformed.keys()), 10)

        bkgdata = np.random.rand(100, 10)
        self.pldamodule.norm(bkgdata, transformed)
        # random input
        for model, modelvec in transformed.iteritems():
            for testuttname, testvec in transformedtest.iteritems():
                score = self.pldamodule.score(model, modelvec, testvec)
                # Should be around this range
                self.assertTrue(-100 <= score <= 100)

    def test_fittransformlarge(self):
        largedata = np.random.rand(10000, 1024)
        labels = np.array([i / 10 for i in range(largedata.shape[0])])
        self.assertIsNone(self.pldamodule.fit(largedata, labels))
        enroledata = np.random.rand(100, 1024)
        enrolelabels = np.array([i % 10 for i in range(enroledata.shape[0])])
        transformed = self.pldamodule.transform(enroledata, enrolelabels)
        testdata = np.random.rand(100, 1024)
        testlabels = np.arange(100)

        transformedtest = self.pldamodule.transform(testdata, testlabels)
        bkgdata = np.random.rand(100, 1024)

        self.pldamodule.norm(bkgdata, transformed)
        for model, modelvec in transformed.iteritems():
            for testuttname, testvec in transformedtest.iteritems():
                score = self.pldamodule.score(model, modelvec, testvec)
                # Should be around this range
                self.assertTrue(-100 <= score <= 100)

    def test_randomtransform(self):
        n_samples = 1938
        m_samples = 969
        enroll_samples = 556
        test_samples = 500
        featdim = 1024
        X = np.random.rand(n_samples, featdim)
        Y = np.array([i / 5 for i in range(n_samples)])
        self.pldamodule.fit(X, Y,2)

        Models_X = np.random.rand(enroll_samples, featdim)
        Models_Y = np.array([i / 4 for i in xrange(enroll_samples)])
        # Starting transformations
        transformed_vectors = self.pldamodule.transform(Models_X, Models_Y)
        # Generate some random background data
        Otherdata = np.random.rand(m_samples, featdim)
        self.pldamodule.norm(Otherdata, transformed_vectors)

        testutt_x = np.random.rand(test_samples, featdim)
        testutt_y = np.arange(test_samples)

        transformedtest_vectors = self.pldamodule.transform(
            testutt_x, testutt_y)

        for model, modelvec in transformed_vectors.iteritems():
            for testutt, testvec in transformedtest_vectors.iteritems():
                score = self.pldamodule.score(model, modelvec, testvec)
                self.assertTrue(-100 <= score <= 100)


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(PLDATest)
    # testResult = unittest.TextTestRunner(verbosity=2).run(suite)
