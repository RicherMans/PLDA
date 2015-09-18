#include "Plda.hpp"

namespace kaldi{
    //Define the static argument first here
    PLDA* PLDA::m_pInstance = NULL;

    PLDA::PLDA():plda(),plda(),stats(),est(stats),adaptor(){
    }

    PLDA* PLDA::getInstance(){
        if (p_mInstance == NULL){
            m_pInstance = new PLDA();
        }
        return m_pInstance;
    }

    void PLDA::addsamples(double weight,const Matrix<double> &group){
        this->stats.AddSamples(weight,group);
    }

    void PLDA::estimate(const PldaConfig &config){
        this->est.Estimate(config,&this->plda);
    }

    //Adapts a trained PLDA model towards some speakers, given as Vector stat
    void PLDA::addadaptstats(double weight,const Vector<double> &stat){
        this->adaptor.AddStats(weight,stat);
    }

    void PLDA::addadaptstats(float weight,const Vector<float> &stat){
        this->adaptor.AddStats(1.0,stat);

    }

    void PLDA::smoothwithincovariance(double factor){
        this->plda.SmoothWithinClassCovariance(factor);
    }

    PLDA::~PLDA(){
        delete est;
        delete m_pInstance;
        delete plda;

    }
    PLDA::sortstats(){
        this->stats.Sort();
    }

}
