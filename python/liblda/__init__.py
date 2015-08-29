from liblda import fitlda,predictlda

class LDA():

    def fit(self,flist,dim):
        self._ldamat = fitlda(flist,dim)

    def predict(self,flist):
        predictlda(flist,self._ldamat)
