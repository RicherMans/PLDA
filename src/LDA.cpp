#include "LDA.hpp"


namespace kaldi{
    // Need to declare the static variable to be visible
    LDA* LDA::m_pInstance = NULL;

    void LDA::accumulate(const VectorBase<BaseFloat> &data, int32 class_id, BaseFloat weight){
        this->Accumulate(data,class_id,weight);
    }

    void LDA::estimate(const LdaEstimateOptions &opts,
                Matrix<BaseFloat> *M,
                Matrix<BaseFloat> *Mfull) const{
        this->Estimate(opts,M,Mfull);
    }

    LDA* LDA::getInstance(){
        if (m_pInstance == NULL){
            m_pInstance = new LDA();
        }
        return m_pInstance;
    }
    void LDA::init(int32 numclasses,int32 featdim){
        this->Init(numclasses,featdim);
    }

    void LDA::getstats(SpMatrix<double> *total_covar,
                SpMatrix<double> *between_covar,
                Vector<double> *total_mean,
                double *sum) const{
        this->GetStats(total_covar,between_covar,total_mean,sum);
    }

    void LDA::getclassmean(Matrix<double> *out_class_mean) const{
        int32 num_class = NumClasses(), dim = Dim();
        out_class_mean->Resize(num_class,dim);
        out_class_mean->CopyFromMat(first_acc_);
          for (int32 c = 0; c < num_class; c++) {
            if (zero_acc_(c) != 0.0) {
              out_class_mean->Row(c).Scale(1.0 / zero_acc_(c));
            }
        }
    }
}
