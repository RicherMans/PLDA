#include <Python.h>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>
#include <algorithm>
//numpy library
#include "numpy/arrayobject.h"
#include <cassert>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

#include "chtk.h"
#include "ivector/plda.h"
#include "kaldi-utils.hpp"

namespace kaldi{


    struct Stats{
        uint32_t size;
        Vector<double> data;
    };



    static PyObject* py_fitplda(PyObject* self,PyObject *args){
        SetVerboseLevel(0);
        //From the given data as features with dimensions (nsamples,featdim ) and labels as (nsamples,)
        //we estimate the statistics.
        //The labels values indicate which label is given for each sample in nsamples
        PyArrayObject* py_inputfeats;
        PyArrayObject* py_labels;
        std::string estimatedmodel;
        uint32_t iters;
        if (! PyArg_ParseTuple( args, "O!O!s", &PyArray_Type,&py_inputfeats,&PyArray_Type,&py_labels,&estimatedmodel,&iters)) return NULL;

        auto n_samples=py_inputfeats->dimensions[0];
        auto featdim =py_inputfeats->dimensions[1];

        assert(py_labels->dimensions[0]==py_inputfeats->dimensions[0]);

        PldaStats stats;

        const Matrix<BaseFloat> &inputfeats = pyarraytomatrix<double>(py_inputfeats);

        long *labels = pyvector_to_type<long>(py_labels);

        std::set<long> u_labels;
        for (auto sample = 0u; sample < n_samples; ++sample) {
            u_labels.insert(labels[sample]);
        }
        auto num_speakers = u_labels.size();
        std::vector<MatrixIndexT> indices[num_speakers];
        for(auto n=0u ; n < n_samples;n++){
            indices[labels[n]].push_back(n);
        }

        for(auto spk=0u; spk < num_speakers;spk ++){
            const Matrix<double> tmp(indices[spk].size(),featdim);
            tmp.CopyRows(inputfeats,indices[m]);
            stats.AddSamples(1.0/indices[m].size(),tmp);
        }

        stats.Sort();

        PldaEstimationConfig config;
        config.num_em_iters = iters;

        PldaEstimator estimator(stats);
        Plda plda;
        estimator.Estimate(config,&plda);

        std::ofstream outf(estimatedmodel);
        plda.Write(outf,true);

        return Py_BuildValue("");
    }

    static PyObject* py_transform(PyObject* self,PyObject* args){

        float smoothfactor;
        PyArrayObject* py_inpututts;
        PyArrayObject* py_labels;
        std::string outputfile;
        std::string pldamodelpath;
        if (! PyArg_ParseTuple( args, "O!O!ss", &PyArray_Type,&py_inpututts,&PyArray_Type,&py_labels,&pldamodelpath,&outputfile)) return NULL;

        std::map<uint32_t,Stats> speakertoutts;
        PyObject *retdict = PyDict_New();

        const Matrix<BaseFloat> &inputfeats = pyarraytomatrix<double>(py_inpututts);
        auto n_samples=py_inputfeats->dimensions[0];
        auto featdim =py_inputfeats->dimensions[1];

        assert(py_labels->dimensions[0]==py_inputfeats->dimensions[0]);
        long *labels = pyvector_to_type<long>(py_labels);

        Plda plda;
        std::ifstream pldain(pldamodelpath);
        plda.Read(pldamodelpath,true);

        for(auto spk = 0u; spk < n_samples;spk++){
            if(speakertoutts.count(spk)==0){
                Stats stat;
                stat.size=0;
                stat.data.Resize(featdim);
                speakertoutts.insert(std::make_pair(spk,stat));
            }
            speakertoutts[labels[spk]].size += 1;
            speakertoutts[labels[spk]].data.AddVecVec(1.0,inputfeats[spk]);
        }

        for(std::map<uint32_t,Stats>::const_iterator it=speakertoutts.begin();it!=speakertoutts.end();it++){
            auto samplesize=it->second.size;
            assert(samplesize>0);
            // Scale the vector by its samplesize to normalize it
            it->second.Scale(1.0/samplesize);
            Vector<double> transformed(featdim);
            plda.TransformIvector(config,it->second,samplesize,&transformed);

            spkid = PyInt_FromSize_t(it->first);
            PyArrayObject* py_transformed = (PyArrayObject* )PyArray_SimpleNewFromData(2,dimensions,NPY_DOUBLE,transformed.Data());
            py_transformed->flags |= NPY_ARRAY_OWNDATA;
            // Set the value in the map as the transformed value
            PyDict_SetItem(retdict,spkid,(PyObject*) py_transformed);
        }

        return Py_BuildValue("O",retdict);


    }

    static PyObject* py_znorm(PyObject* self, PyObject* args){


    }


    static PyMethodDef libpldaModule_methods[]={
        {"fitplda",py_fitplda,METH_VARARGS},
        {"transform",py_transform,METH_VARARGS},
        {NULL,NULL}
    };

    extern "C" void initlibplda()
    {
        (void) Py_InitModule("libplda",libpldaModule_methods);

        import_array();
    }

}
