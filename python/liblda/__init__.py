from liblda import fitlda,predictlda

# the LDA class can be used by simply initzialize it and use the fit and predict parameters
class LDA():

    def fit(self,flist,dim):
        '''
        Estimates an LDA matrix from the given flist and transforms that matrix to dimension dim
        flist is a dict where the keys are the given speakers ( how many classes there are  ) and
        the values correspond to the HTK type features as paths
        '''
        self._ldamat = fitlda(flist,dim)

    def predict(self,featurefile):
        '''
        Predicts for the given file given as f
        '''
        if self._ldamat is None:
            raise ValueError("THe method .fit needs to be called before predict!")
        return predictlda(featurefile,self._ldamat)
