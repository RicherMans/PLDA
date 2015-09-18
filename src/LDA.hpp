#ifndef LDA_HPP_
#define LDA_HPP_

#include "transform/lda-estimate.h"

namespace kaldi{

// CLass is implemented as singleton
class LDA: private LdaEstimate {
private:
    static LDA* m_pInstance;
    LDA()=default;
    LDA(const LDA &other) = delete;
    LDA(const LDA &&other ) = delete;
public:
    static LDA* getInstance();
    // Just a forward on the Kaldi method
    void accumulate(const VectorBase<BaseFloat> &data, int32 class_id, BaseFloat weight = 1.0);
    // Estimate the LDA transform. THe matrix M is returned which is the transformation matrix for the features
    void estimate(const LdaEstimateOptions &opts,
                Matrix<BaseFloat> *M,
                Matrix<BaseFloat> *Mfull = NULL) const;
    // Initalize the model, for a given number of classes and feature dims
    void init(int32 numclasses,int32 featdim);
    // Get the current stats of the LDA model
    void getstats(SpMatrix<double> *total_covar,
                SpMatrix<double> *between_covar,
                Vector<double> *total_mean,
                double *sum) const;
    // Fills in the given matrix class_mean with the means of all classes.
    void getclassmean(Matrix<double> *class_mean)const;
};
}

#endif // LDA_HPP_
