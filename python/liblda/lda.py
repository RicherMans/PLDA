import numpy as np
# from liblda import MLDA
from scipy.linalg import eigh
from scipy.sparse import issparse
from scipy.misc import logsumexp
# the LDA class can be used by simply initzialize it and use the fit and
# predict parameters


def _class_cov(X, y, priors=None, shrinkage=None):
    classes = np.unique(y)
    covs = []
    for group in classes:
        Xg = X[y == group,:]
        covs.append(np.atleast_2d(empirical_covariance(Xg, shrinkage)))
    return np.average(covs, axis=0, weights=priors)


def empirical_covariance(X, assume_centered=False):
    """Computes the Maximum likelihood covariance estimator
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data from which to compute the covariance estimate
    assume_centered : Boolean
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.
    Returns
    -------
    covariance : 2D ndarray, shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        print("Only one sample available. "
                "You may want to reshape your data array")

        if assume_centered:
            covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like, shape (n_features,)
        Class means.
    """
    means = []
    classes = np.unique(y)
    for group in classes:
        Xg = X[y == group,:]
        means.append(Xg.mean(0))
    return np.asarray(means)


def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly
    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.
    """
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


class LDA():

    def __init__(self, solver='svd', priors=None):
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

    # Fit function is completely based on SKlearns LDA, see
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/discriminant_analysis.py
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
        self._classes = np.unique(labels)
        if self.priors is None:  # estimate priors from sample
            # non-negative ints
            _, y_t = np.unique(labels, return_inverse=True)
            self.priors = np.bincount(y_t) / float(len(labels))
        else:
            self.priors = np.asarray(self.priors)

        if self.priors.sum() != 1:
            self.priors = self.priors / self.priors.sum()

        # Accumulate the statistics
        # super(LDA, self).fit(features, labels)

        # Run the least squares solver
        if self.solver == 'lsqr':
            self._solve_lsqr(features, labels)
        elif self.solver == 'svd':
            self._solve_svd(features, labels)
        elif self.solver == 'eigen':
            self._solve_eigen(features,labels)

    def _solve_eigen(self, X, y):
        """Eigenvalue solver.
        The eigenvalue solver computes the optimal solution of the Rayleigh
        coefficient (basically the ratio of between class scatter to within
        class scatter). This solver supports both classification and
        dimensionality reduction .
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        Notes
        -----
        This solver is based on [1]_, section 3.8.3, pp. 121-124.
        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        self._means = _class_means(X, y)
        withincov = _class_cov(X, y, self.priors)

        Sw = withincov  # within scatter
        St = empirical_covariance(X)  # total scatter
        Sb = St - Sw  # between scatter
        evals, evecs = eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1]
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        # evecs /= np.linalg.norm(evecs, axis=0)  # doesn't work with numpy 1.6
        evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)

        self._scalings = evecs
        self._coef = np.dot(self._means, evecs).dot(evecs.T)
        self._intercept = (-0.5 * np.diag(np.dot(self._means, self._coef.T))
                + np.log(self.priors))

    def _solve_svd(self, X, y):
        n_samples, featdim = X.shape
        n_classes = len(self._classes)
        tol = 1e-4

        self._means = _class_means(X, y)
        xc = []
        for ind, group in enumerate(self._classes):
            feature = X[y == group,:]
            xc.append(feature - self._means[ind])

        self._xbar = np.dot(self.priors, self._means)
        # Generate a matrix from the zero mean vectors
        xc = np.concatenate(xc, axis=0)

        # 1) within (univariate) scaling by with classes std-dev
        #
        stddev = xc.std(axis=0)
        stddev[stddev == 0] = 1.
        fac = 1. / (n_samples - n_classes)

        # 2) Within variance scaling
        X = np.sqrt(fac) * (xc / stddev)
        _, S, V = np.linalg.svd(X, full_matrices=False)
        rank = np.sum(S > tol)
        scalings = ((V[:rank] / stddev).T / S[:rank])

        # 3) Between variance scaling
        # Scale weighted centers
        X = np.dot(((np.sqrt((n_samples * self.priors) * fac)) *
            (self._means - self._xbar).T).T, scalings)

        _, S, V = np.linalg.svd(X, full_matrices=0)

        rank = np.sum(S > tol * S[0])
        self._scalings = np.dot(scalings, V.T[:, :rank])

        coef = np.dot(self._means - self._xbar, self._scalings)
        self._intercept = (-0.5 * np.sum(coef ** 2, axis=1)
                + np.log(self.priors))

        self._coef = np.dot(coef, self._scalings.T)

        self._intercept -= np.dot(self._xbar, self._coef.T)

    def _solve_lsqr(self, X, y):
        """Least squares solver.
        The least squares solver computes a straightforward solution of the
        optimal decision rule based directly on the discriminant functions. It
        can only be used for classification (with optional shrinkage), because
        estimation of eigenvectors is not performed. Therefore, dimensionality
        reduction with the transform is not supported.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        shrinkage : string or float, optional
        Notes
        -----
        This solver is based on [1]_, section 2.6.2, pp. 39-41.
        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        self._means = _class_means(X, y)
        # Get the class covaraince from SKlearns approach ...
        # Kaldi's estimation using between covar somehow doesnt work out
        cov = _class_cov(X, y, self.priors)
        self._coef = np.linalg.lstsq(
                cov, self._means.T)[0].T
        self._intercept = (-0.5 * np.diag(np.dot(self._means, self._coef.T))
                + np.log(self.priors))

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
            raise ValueError("This %(name)s instance is not fitted yet" % {'name': type(self).__name__})

        n_features = self._coef.shape[1]

        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                    % (X.shape[1], n_features))

        scores = safe_sparse_dot(X, self._coef.T, True) + self._intercept
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict_proba(self, sample):
        '''
        Function: predict_proba
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
        if len(self._classes) == 2:  # binary case
            return np.column_stack([1 - prob, prob])
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob

    def predict_log_proba(self, sample):
        '''
        Function: predict_log_proba
        Summary: The same function as predict_prob, but returns the log probability
        Examples:
        Attributes:
            @param (self):
            @param (sample):A 2 dimensional array in the shape of {nsamples,nfeatures}
        Returns: the log probability of the sample belonging to the classes
            array, shape (n_samples, n_classes)
            Estimated log probabilities.
        '''
        # values = self.decision_function(sample)
        # llk = (values - values.max(axis=1)[:, np.newaxis])
        # normalizationconstant = logsumexp(llk, axis=1)
        # return llk - normalizationconstant[:, np.newaxis]
        values = self.decision_function(sample)
        llk = (values - values.max(axis=1)[:, np.newaxis])
        normalizationconstant = logsumexp(llk, axis=1)
        return llk - normalizationconstant[:, np.newaxis]


    def transform(self, X,n_components=None):
        """Project data to maximize class separation.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed data.
        """
        if self.solver == 'lsqr':
            raise NotImplementedError("transform not implemented for 'lsqr' "
                    "solver (use 'svd' or 'eigen').")

            if self.solver == 'svd':
                X_new = np.dot(X - self._xbar, self._scalings)
        elif self.solver == 'eigen':
            X_new = np.dot(X, self._scalings)
        n_components = X.shape[1] if n_components is None \
                else n_components
        return X_new[:, :n_components]
