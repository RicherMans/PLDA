import unittest
import numpy as np
from liblda import LDA


class LDATest(unittest.TestCase):

    def setUp(self):
        self.data = np.random.rand(2000, 10)
        self.labels = np.array([i % 10 for i in range(self.data.shape[0])])

    def test_fitpredictsvd(self):
        ldamodule = LDA(solver='svd')
        self.assertIsNone(ldamodule.fit(self.data, self.labels))
        testdata = np.random.rand(100, 10)
        # testlabels = np.array([i%10 for i in range(testdata.shape[0])])
        scores = ldamodule.predict_log_prob(testdata)
        for scoreset in scores:
            for score in scoreset:
                self.assertLessEqual(score, 0)

    def test_fitpredictlsqr(self):
        ldamodule = LDA(solver='lsqr')
        self.assertIsNone(ldamodule.fit(self.data, self.labels))
        testdata = np.random.rand(100, 10)
        # testlabels = np.array([i%10 for i in range(testdata.shape[0])])
        scores = ldamodule.predict_log_prob(testdata)
        for scoreset in scores:
            for score in scoreset:
                self.assertLessEqual(score, 0)

    def test_fitpredicteigen(self):
        ldamodule = LDA(solver='eigen')
        self.assertIsNone(ldamodule.fit(self.data, self.labels))
        testdata = np.random.rand(100, 10)
        # testlabels = np.array([i%10 for i in range(testdata.shape[0])])
        scores = ldamodule.predict_log_prob(testdata)
        for scoreset in scores:
            for score in scoreset:
                self.assertLessEqual(score, 0)

    def test_transform(self):
        ldamodule = LDA(solver='svd')
        self.assertIsNone(ldamodule.fit(self.data, self.labels))

        totransform = np.random.rand(100, 10)
        transformed = ldamodule.transform(totransform, 5)
        self.assertEqual(transformed.shape, (100, 5))


if __name__ == '__main__':
    unittest.main()
