import numpy as np
from scipy.misc import logsumexp
from liblda import fitlda, predictldafromutterance, predictldafromarray, getstats, getclassmeans
# the LDA class can be used by simply initzialize it and use the fit and
# predict parameters


class LDA():

    def __init__(self, priors=None, solver='eigen'):
        '''
        Function: __init__
        Summary: Inits an LDA object
        Examples:
        Attributes:
            @param (self):
            @param (priors) default=None: Priors of the given dataset. If None, they will be sampled by the given classset
            @param (solver) default='eigen': If solver is eigen, KALDI lda estimator is used ( the default one ), if solver is 'lsqr'
            we do the least squares estimation and cannot transform the given features!
        '''
        self.priors = priors
        self.solver = solver

    def fit(self, flist, dim):
        '''
        Estimates an LDA matrix from the given flist and transforms that matrix to dimension dim
        flist is a dict where the keys are the given speakers ( how many classes there are  ) and
        the values correspond to the HTK type features as paths
        '''
        # Estimate the lda transformation matrix and also get the bins for
        # estimating the priors
        self._ldamat, _bins = fitlda(flist, dim, transform=False)
        # Get the statistics from the fitted model
        _, self._tot_cov, self._bet_cov, self._n_samples = getstats()
        self._means = getclassmeans()
        self._covariance = self._tot_cov - self._bet_cov
        # print self._means
        if self.priors is None:
            _, self._bins = np.unique(_bins, return_inverse=True)
            self.priors_ = np.bincount(self._bins) / float(self._n_samples)
        # self.covariance_ = _class_cov(X, y, self.priors_)
        if self.solver == 'lsqr':
            self._leastsquares()

    def _getstats(self):
        return getstats()

    def _leastsquares(self):
        if self._ldamat is None:
            raise ValueError(
                "THe method .fit needs to be called before getting any statistics!")
        self._coef = np.linalg.lstsq(self._covariance, self._means.T)[0].T
        self.intercept_ = (-0.5 * np.diag(np.dot(self._means, self._coef.T))
                           + np.log(self.priors_))

    def decision_function(self, X):
        """Predict confidence scores for samples.
        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        if not hasattr(self, '_coef') or self._coef is None:
            raise ValueError("This %(name)s instance is not fitted"
                             "yet" % {'name': type(self).__name__})

        # X = check_array(X, accept_sparse='csr')

        n_features = self._coef.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        scores = np.dot(X, self._coef.T) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict_prob(self, sample):
        '''
        Function: predict_prob
        Summary: Predicts the probability of the given sample. Previously you need have to run leastsquares or some other fitting method
        Examples:
        Attributes:
            @param (self):
            @param (sample):A 2 dimensional array in the shape of {nsamples,nfeatures}
        Returns: a normalized score for the given sample for each class
        array, shape (n_samples, n_classes)
            Estimated log probabilities.
        '''

        prob = self.decision_function(sample)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        # OvR normalization, like LibLinear's predict_probability
        prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
        return prob

    def predict_log_prob(self, sample):
        '''
        Function: predict_log_prob
        Summary: The same function as predict_prob, but returns the log probability
        Examples:
        Attributes:
            @param (self):
            @param (sample):A 2 dimensional array in the shape of {nsamples,nfeatures}
        Returns: the log probability of the sample belonging to the classes
            array, shape (n_samples, n_classes)
            Estimated log probabilities.
        '''
        values = self.decision_function(sample)
        llk = (values - values.max(axis=1)[:, np.newaxis])
        normalizationconstant = logsumexp(llk, axis=1)
        return llk - normalizationconstant[:, np.newaxis]

    def transform(self, featurefile):
        '''
        Predicts for the given file given as f
        '''
        if self._ldamat is None or len(self._ldamat) == 0:
            raise ValueError(
                "THe method .fit needs to be called before predict (with parameter transform = true)!")
        return predictldafromutterance(featurefile, self._ldamat)

    def transformmat(self, featuremat):
        if self._ldamat is None or len(self._ldamat) == 0:
            raise ValueError(
                "THe method .fit needs to be called before predict (with parameter transform = true)!")
        return predictldafromarray(featuremat, self._ldamat)
