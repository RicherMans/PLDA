#ifndef COVARIANCESTATS_H_
#define COVARIANCESTATS_H_
#endif

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "thread/kaldi-task-sequence.h"

namespace kaldi{

    class CovarianceStats
    {
    public:
        CovarianceStats (int32 dim):tot_covar_(dim),between_covar_(dim),num_spk_(0),num_utt_(0) {};
        virtual ~CovarianceStats ();

    private:
        SpMatrix<double> tot_covar_;
        SpMatrix<double> between_covar_;
        int32 num_spk_;
        int32 num_utt_;
    };
}
