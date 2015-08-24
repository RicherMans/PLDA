#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include "transform/lda-estimate.h"
#include "CovarianceStats.hpp"
#include "chtk.h"

namespace kaldi{


template<typename T>
std::map<std::string,T> pythondict_to_map(){

    }


static PyObject* py_fitlda(PyObject* self,PyObject* args){
    PyObject* dict;
    PyObject* ldaopts;
    /* the O! parses for a Python object (listObj) checked
     *    to be of type PyList_Type */
    if (! PyArg_ParseTuple( args, "O!|s", &PyDict_Type, &dict,&ldaopts)) return NULL;
    //Check if the given argument is really a dict
    PyDict_Check(dict);
    auto numitems = PyDict_Size(dict);
    //equal to dict.items()
    PyObject* items = PyDict_Items(dict);
    //Variable to check if the sizes of the utterance for every key are consistent
    auto check_utts=0;
    LdaEstimate lda;
    for (auto i = 0u; i < numitems; ++i) {
        PyObject* item = PyList_GetItem(items,i);
        PyObject* spk=PyTuple_GetItem(item,0);
        PyObject* values=PyTuple_GetItem(item,1);
        PyList_Check(values);
        auto num_utts=PyList_Size(values);
        //if (i==0) {
            //check_utts = num_utts;
        //} else {
            //if (check_utts != num_utts) {
                //return PyErr_Format(PyExc_ValueError,"Size of lists inconsistent!");
            //}
        //}
        if (num_utts < 0){
            std::string err("Values need to be a list of features(string)!");
            return PyErr_Format(PyExc_ValueError,err.c_str());
        }
        //We assume that we have key value pairs, where the values are a list of strings representing
        //The utterances for the speaker
        for (auto j = 0; j < num_utts; ++j) {
            PyObject *utt = PyList_GetItem(values,j);
            PyString_Check(utt);
            const char* curvalue = PyString_AsString(utt);

            try{
                //Reads in the feature from the given file. We do not extend the feature ( hence the paremter 0 )
                chtk::htkarray feature = chtk::htk_load(std::string(curvalue),0);
                //Samplesize is in bytes, so we need to divide it by 4 to get the actual dimension
                auto featdim = feature.samplesize/4;
                if(lda.Dim()==0){
                    lda.Init(numitems,featdim);
                }
                //feat is just an temporarary vector , which is refilled for every sample
                Vector<BaseFloat> feat(featdim);
                std::vector<float> read_feats= feature.as_vec<float>();
                //Copy the data of every sample into the Vector so that Kaldi can use it
                for (auto k = 0u; k < read_feats.size(); k+=featdim) {
                    std::copy(read_feats.begin()+k,read_feats.begin()+k+featdim,feat.Data());
                    lda.Accumulate(feat,i);
                }
            }catch(const std::exception &e){
                std::cerr << e.what();
                return PyErr_Format(PyExc_OSError,e.what());
            }
        }
    }
    std::cerr << lda.NumClasses() << std::endl;
    std::cerr << lda.Dim() << std::endl;
    LdaEstimateOptions opts;
    opts.dim = 1;
    Matrix<BaseFloat> lda_mat;
    Matrix<BaseFloat> full_lda_mat;
    lda.Estimate(opts,&lda_mat,&full_lda_mat);
    std::cerr << lda_mat << std::endl;
    std::cerr << lda_mat.NumRows() << " " << lda_mat.NumCols() <<std::endl;

    return Py_BuildValue("");
}

/*
 *  * Bind Python function names to our C functions
 *   */
static PyMethodDef libldaModule_methods[] = {
    {"fitlda",py_fitlda,METH_VARARGS},
    {NULL, NULL}
};

/*
 *  * Python calls this to let us initialize our module
 *   */
extern "C" void initliblda()
{
    (void) Py_InitModule("liblda", libldaModule_methods);

}
}
