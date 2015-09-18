import numpy as np
from scipy.misc import logsumexp
from liblda import predictldafromutterance, predictldafromarray, getstats, getclassmeans, fitldafromdata, estimate
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

    def fit(self, features, labels):
        '''
        Function: fit
        Summary: Estimates the statistics needed to transform or to do any decisions for the given dataset in features
        Examples:
        Attributes:
            @param (self):
            @param (features):np.array with dimensions (n_samples,feat_dim), which will be used to estimate the statistics
            @param (labels):np.array with dimensions (n_samples,) , where each value represents a speaker id, corresponding to the
            features array!
        Returns: None
        '''
        # Accumulate the statistics
        fitldafromdata(features, labels)
        # Get the statistics from the fitted model
        _, self._tot_cov, self._bet_cov, self._n_samples = getstats()
        self._means = getclassmeans()
        self._covariance = self._tot_cov - self._bet_cov
        # print self._covariance
        # self._covariance = _class_cov(features, labels)
        n_samples = features.shape[0]
        # print self._covariance
        # print self._means
        if self.priors is None:
            _, self._bins = np.unique(labels, return_inverse=True)
            self.priors_ = np.bincount(self._bins) / float(n_samples)
        if self.solver == 'lsqr':
            self._leastsquares()

    def _getstats(self):
        return getstats()

    def _leastsquares(self):
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

    def transform(self, featurefile, targetdim):
        '''
        Predicts for the given file given as f
        '''
        ldamat = estimate(targetdim)
        return predictldafromutterance(featurefile, ldamat)

    def transformmat(self, featuremat, targetdim):
        ldamat = estimate(targetdim)
        return predictldafromarray(featuremat, ldamat)
