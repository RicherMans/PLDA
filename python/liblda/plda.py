from libplda import MPlda


class PLDA(object):

    def __init__(self):
        self._instance = MPlda()

    def fit(self, x, y, iters=10):
        return self._instance.fit(x, y, iters)

    def transform(self, x, y):
        '''
        Function: transform
        Summary: Transforms a given set of X vectors and Y labels to the PLDA dimension
        Examples:
        Attributes:
            @param (self):
            @param (x):The vectors which need to be transformed
            @param (y):Representing the labels for each vector
        Returns: A transformed vector
        '''
        return self._instance.transform(x, y)

    def norm(self, vectors, transformedvecs, numutts=0):
        '''
        Function: norm
        Summary: Normalizes the given model with mean/variance estimators
        Examples:
        Attributes:
            @param (self):
            @param (vectors):The input vectors for accumulating the statistics
            @param (transformedvecs):The enrol models which are scored against
            @param (numutts) default=0: Number of utterances which are considered
        Returns: None
        '''
        return self._instance.norm(vectors, transformedvecs, numutts)

    def score(self, target, xvec, yvec):
        '''
        Function: score
        Summary: Scores a given target against a given x,y vector pair
        Examples:
        Attributes:
            @param (self):InsertHere
            @param (target):The id of the fitted target vector
            @param (xvec):The enrolment vector
            @param (yvec):The test vector
        Returns: A score ( float)
        '''
        return self._instance.score(target, xvec, yvec)
