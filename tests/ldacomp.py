import unittest
import numpy as np
from liblda import LDA
from sklearn.lda import LDA as SKLDA
import sklearn.lda


class LDATest(unittest.TestCase):

    def setUp(self):
        self.ldamodule1 = LDA('svd')
        self.ldamodule2 = SKLDA()
        self.data = np.random.rand(10, 10)
        self.labels = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])

    def test_fitpredict(self):
        np.set_printoptions(precision=4)
        self.ldamodule1.fit(self.data, self.labels)
        self.ldamodule2.fit(self.data, self.labels)
        testdata = np.random.rand(10, 10)
        # testlabels = np.array([i%10 for i in range(testdata.shape[0])])
        scores = self.ldamodule1.predict_log_prob(testdata)
        # scores1 = self.ldamodule2.predict_log_proba(testdata)

        # for scoreset1, scoreset2 in zip(scores, scores1):
        #     for score1, score2 in zip(scoreset1, scoreset2):
        #         # print(score1,score2)
        #         self.assertAlmostEqual(score1, score2, 2)


if __name__ == '__main__':
    unittest.main()
