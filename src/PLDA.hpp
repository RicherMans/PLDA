#ifndef PLDA_HPP_
#define PLDA_HPP_
#include "ivector/plda.h"


namespace kaldi{

    // CLass is implemented as singleton
    class PLDA{
        private:
            static PLDA* m_pInstance;
            Plda plda;
            PldaEstimator est;
            PldaStats stats;
            PldaUnsupervisedAdaptor adaptor;
            PLDA();
            PLDA(const PLDA& other) = delete;
            PLDA(const PLDA&& other) = delete;
            ~PLDA();
        public:
            void addsamples(double weight,const Matrix<double> &group);
            PLDA* getInstance();
            void sortstats();
            void estimate(const PldaConfig &config);
            void addadaptstats(double weight,const Vector<double> &stat);
            void addadaptstats(float weight,const Vector<float> &stat);
            void smoothwithincovariance();
    }
}
#endif
